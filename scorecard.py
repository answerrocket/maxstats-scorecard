from __future__ import annotations
from types import SimpleNamespace

from skill_framework import SkillVisualization, skill, SkillParameter, SkillInput, SkillOutput
from skill_framework.preview import preview_skill
from skill_framework.skills import ExportData
from skill_framework.layouts import wire_layout

from answer_rocket import AnswerRocketClient
from ar_analytics.helpers.utils import SkillPlatform

import pandas as pd
import json
import jinja2
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Load templates from resources
TABLE_TEMPLATE_PATH = Path(__file__).parent / "resources" / "TABLE_TEMPLATE.txt"
CHART_TEMPLATE_PATH = Path(__file__).parent / "resources" / "CHART_TEMPLATE.txt"

with open(TABLE_TEMPLATE_PATH, 'r') as f:
    TABLE_TEMPLATE = f.read()

with open(CHART_TEMPLATE_PATH, 'r') as f:
    CHART_TEMPLATE = f.read()

@skill(
    name="scorecard",
    description="KPI Scorecard for Max Product Performance showing user activity metrics",
    parameters=[
        SkillParameter(
            name="tenants",
            is_multi=True,
            constrained_to="enums",
            description="Tenant names to analyze",
            constrained_values=['abibg prod', 'cpw prod', 'suntorymax prod'],
            default_value="cpw prod"
        )
    ]
)
def scorecard(parameters: SkillInput) -> SkillOutput:
    print(f"Skill received parameters: {parameters.arguments}")

    # Get parameters
    tenants = parameters.arguments.tenants if hasattr(parameters.arguments, 'tenants') else ['cpw prod']
    if isinstance(tenants, str):
        tenants = [tenants]

    # Get database connection info
    sp = SkillPlatform()
    dataset_metadata = sp.data.get_metadata()

    # Execute SQL query
    user_stats = get_users_stats(tenants, dataset_metadata)

    export_data = []

    # Generate all data first
    monthly_cohort = create_monthly_cohort_data(user_stats)
    quarterly_cohort = create_quarterly_cohort_data(user_stats)
    rmau = create_monthly_returning_users(user_stats)
    rqau = create_quarterly_returning_users(user_stats)
    active_users = get_users_last_90_days(user_stats)
    rolling_90 = create_90_day_rolling_averages(user_stats)
    rolling_7 = create_7_day_rolling_averages(user_stats)

    # Prepare export data
    export_data.append(ExportData(name="Monthly Cohort", data=monthly_cohort))
    export_data.append(ExportData(name="Quarterly Cohort", data=quarterly_cohort))
    export_data.append(ExportData(name="RMAU", data=rmau))
    export_data.append(ExportData(name="RQAU", data=rqau))
    export_data.append(ExportData(name="Active Users", data=active_users))

    # Create visualizations using render_layout
    visualizations = render_layout(
        monthly_cohort=monthly_cohort,
        quarterly_cohort=quarterly_cohort,
        rmau=rmau,
        rqau=rqau,
        active_users=active_users,
        rolling_90=rolling_90,
        rolling_7=rolling_7,
        tenants=tenants
    )
    final_prompt = f"""Use 4-5 bullet points to highlight the key metrics for tenants {', '.join(tenants)}:
    monthly_cohort: {monthly_cohort}
    quarterly_cohort: {quarterly_cohort}
    rmau: {rmau}
    rqau: {rqau}
    active_users: {active_users}
    rolling_90: {rolling_90}
    rolling_7: {rolling_7}
    """

    return SkillOutput(
        visualizations=visualizations,
        export_data=export_data,
        final_prompt=final_prompt
    )


def get_users_stats(tenants, dataset_metadata):
    """Query consolidated user activity data from database"""
    # Build the tenant filter list
    tenant_filters = ', '.join([f"'{t.lower()}'" for t in tenants])
    database_id = dataset_metadata.get("database_id")
    dataset_source = dataset_metadata.get("sql_table")
    

    # SQL query based on the original JSON sequential skill
    query = f"""
    SELECT
        user_name,
        asked_at_timestamp AS question_date,
        COUNT(*) AS daily_questions,
        tenant_name || ' ' ||
        CASE
            WHEN tenant_host IS NULL OR tenant_host = '' THEN ''
            WHEN POSITION('.' IN tenant_host) = 0 THEN ''
            WHEN POSITION('.' IN tenant_host) = LENGTH(tenant_host) THEN ''
            WHEN POSITION('.' IN SUBSTRING(tenant_host FROM POSITION('.' IN tenant_host) + 1)) = 0 THEN ''
            ELSE SUBSTRING(
                tenant_host
                FROM POSITION('.' IN tenant_host) + 1
                FOR POSITION('.' IN SUBSTRING(tenant_host FROM POSITION('.' IN tenant_host) + 1)) - 1
            )
        END AS tenant,
        CASE WHEN asked_at_timestamp >= CURRENT_DATE - INTERVAL '90 days' THEN 1 ELSE 0 END AS recent_90_days,
        CASE WHEN asked_at_timestamp >= CURRENT_DATE - INTERVAL '7 days' THEN 1 ELSE 0 END AS recent_7_days,
        CASE WHEN asked_at_timestamp::date >= CURRENT_DATE - INTERVAL '1 day'
             AND asked_at_timestamp::date < CURRENT_DATE THEN 1 ELSE 0 END AS new_user_yesterday,
        MIN(asked_at_timestamp) OVER (PARTITION BY user_name) AS user_first_seen
    FROM {dataset_source}
    WHERE LOWER(tenant) IN ({tenant_filters})
    AND asked_at_timestamp IS NOT NULL
    AND LOWER(user_name) NOT LIKE '%answerrocket%'
    GROUP BY user_name, asked_at_timestamp, tenant_name, tenant_host
    """

    # Execute query using AnswerRocketClient
    ar_client = AnswerRocketClient()
    sql_result = ar_client.data.execute_sql_query(database_id, query, row_limit=1000000)
    print('executed query', query)
    print('sql_result', sql_result)
    if sql_result.success:
        df = sql_result.df
        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        logger.info(f"Successfully fetched {len(df)} rows of user stats data")
        return df
    else:
        logger.error(f"Error executing SQL query: {sql_result.error if hasattr(sql_result, 'error') else 'Unknown error'}")
        return pd.DataFrame()


def create_monthly_cohort_data(df):
    """Create monthly cohort analysis from consolidated user stats"""
    if df.empty:
        return pd.DataFrame()

    # Create cohort data from base dataframe
    cohort_data = []
    for _, row in df.iterrows():
        user_first_month = pd.to_datetime(row['user_first_seen']).strftime('%Y-%m')
        activity_month = pd.to_datetime(row['question_date']).to_period('M').strftime('%Y-%m')

        cohort_data.append({
            'cohort': user_first_month,
            'month': activity_month,
            'user_name': row['user_name'],
            'tenant': row['tenant']
        })

    cohort_df = pd.DataFrame(cohort_data)

    cohort_df['unique_user'] = cohort_df['tenant'].astype(str) + '_' + cohort_df['user_name']
    # Count returning users by cohort and month
    cohort_summary = cohort_df.groupby(['cohort', 'month'])['unique_user'].nunique().reset_index()
    cohort_summary.rename(columns={'unique_user': 'returning_users'}, inplace=True)

    # Create pivot table
    df_pivot = cohort_summary.pivot_table(index='cohort', columns='month', values='returning_users', fill_value=0)

    # Convert index and columns to datetime to ensure proper calculations
    df_pivot.index = pd.to_datetime(df_pivot.index)
    df_pivot.columns = pd.to_datetime(df_pivot.columns)
    def calculate_month_diff(row):
        cohort_start = row.name
        return row.index.to_series().apply(lambda x: (x.year - cohort_start.year) * 12 + (x.month - cohort_start.month))
    # Apply the function to calculate month differences
    month_diffs = df_pivot.apply(calculate_month_diff, axis=1)
    # Reorganize the data based on month differences
    target_view = pd.DataFrame(index=df_pivot.index)
    # Populate the new DataFrame based on month differences
    for i in range(len(df_pivot.columns)):
        target_view[f'Month {i}'] = month_diffs.apply(lambda x: df_pivot.loc[x.name][x == i].iloc[0] if not df_pivot.loc[x.name][x == i].empty else '', axis=1)
    # Convert index back to the period format for display
    target_view.index = target_view.index.to_period('M')
    target_view = target_view.sort_index(ascending=False)
    # Convert all columns to numeric, setting errors='coerce' will convert non-numeric values to NaN
    df_numeric = target_view.apply(pd.to_numeric, errors='coerce')
    # Fill NaN values with 0
    df_filled = df_numeric.fillna(0)
    # Calculate the sum for each column in the DataFrame with NaN values replaced by 0
    totals = df_filled.sum(axis=0)
    print(totals)
    # Append the totals row to the original DataFrame
    target_view.loc['Total'] = totals

    # Convert Period index to string for JSON serialization
    target_view.index = target_view.index.astype(str)

    # Convert all numeric types to native Python types for JSON serialization
    # Replace 0s with empty strings for cleaner display
    for col in target_view.columns:
        target_view[col] = target_view[col].apply(lambda x: int(x) if pd.notnull(x) and x != '' and x != 0 else ('' if x == 0 else x))

    return target_view


def create_quarterly_cohort_data(df):
    """Create quarterly cohort analysis from consolidated user stats"""
    if df.empty:
        return pd.DataFrame()

    # Create cohort data from base dataframe
    cohort_data = []
    for _, row in df.iterrows():
        user_first_quarter = pd.to_datetime(row['user_first_seen']).to_period('Q').strftime('%Y-Q%q')
        activity_quarter = pd.to_datetime(row['question_date']).to_period('Q').strftime('%Y-Q%q')

        cohort_data.append({
            'cohort': user_first_quarter,
            'quarter': activity_quarter,
            'user_name': row['user_name'],
            'tenant': row['tenant']
        })

    cohort_df = pd.DataFrame(cohort_data)
    cohort_df['unique_user'] = cohort_df['tenant'].astype(str) + '_' + cohort_df['user_name']
    # Count returning users by cohort and quarter
    cohort_summary = cohort_df.groupby(['cohort', 'quarter'])['unique_user'].nunique().reset_index()
    cohort_summary.rename(columns={'unique_user': 'returning_users'}, inplace=True)


    # Create pivot table
    df_pivot = cohort_summary.pivot_table(index='cohort', columns='quarter', values='returning_users', fill_value=0)

    # Convert index and columns to datetime to ensure proper calculations
    df_pivot.index = pd.to_datetime(df_pivot.index)
    df_pivot.columns = pd.to_datetime(df_pivot.columns)

    def calculate_quarter_diff(row):
        cohort_start = row.name
        return row.index.to_series().apply(lambda x: (x.year - cohort_start.year) * 4 + (x.quarter - cohort_start.quarter))

    # Apply the function to calculate quarter differences
    quarter_diffs = df_pivot.apply(calculate_quarter_diff, axis=1)

    # Reorganize the data based on quarter differences
    target_view = pd.DataFrame(index=df_pivot.index)

    # Populate the new DataFrame based on quarter differences
    for i in range(len(df_pivot.columns)):
        target_view[f'Quarter {i}'] = quarter_diffs.apply(lambda x: df_pivot.loc[x.name][x == i].iloc[0] if not df_pivot.loc[x.name][x == i].empty else '', axis=1)

    # Convert index back to the period format for display
    target_view.index = target_view.index.to_period('Q')
    target_view = target_view.sort_index(ascending=False)

    # Convert all columns to numeric, setting errors='coerce' will convert non-numeric values to NaN
    df_numeric = target_view.apply(pd.to_numeric, errors='coerce')

    # Fill NaN values with 0
    df_filled = df_numeric.fillna(0)

    # Calculate the sum for each column in the DataFrame with NaN values replaced by 0
    totals = df_filled.sum(axis=0)
    print(totals)
    # Append the totals row to the original DataFrame
    target_view.loc['Total'] = totals

    # Convert Period index to string for JSON serialization
    target_view.index = target_view.index.astype(str)

    # Convert all numeric types to native Python types for JSON serialization
    # Replace 0s with empty strings for cleaner display
    for col in target_view.columns:
        target_view[col] = target_view[col].apply(lambda x: int(x) if pd.notnull(x) and x != '' and x != 0 else ('' if x == 0 else x))

    return target_view


def create_monthly_returning_users(df):
    """Calculate monthly returning users (RMAU)"""
    if df.empty:
        return pd.DataFrame()

    df['year_month'] = pd.to_datetime(df['question_date']).dt.to_period('M').astype(str)
    monthly_data = df.groupby(['year_month', 'user_name']).agg({
        'daily_questions': 'sum'
    }).reset_index()

    users_asking = monthly_data.groupby('year_month')['user_name'].nunique().reset_index()
    users_asking.rename(columns={'user_name': 'MAU'}, inplace=True)

    # Sort by year_month in descending order (most recent first)
    users_asking = users_asking.sort_values('year_month', ascending=False)

    return users_asking


def create_quarterly_returning_users(df):
    """Calculate quarterly returning users (RQAU)"""
    if df.empty:
        return pd.DataFrame()

    df['year_quarter'] = pd.to_datetime(df['question_date']).dt.to_period('Q').astype(str)
    quarterly_data = df.groupby(['year_quarter', 'user_name']).agg({
        'daily_questions': 'sum'
    }).reset_index()

    users_asking = quarterly_data.groupby('year_quarter')['user_name'].nunique().reset_index()
    users_asking.rename(columns={'user_name': 'QAU'}, inplace=True)

    # Sort by year_quarter in descending order (most recent first)
    users_asking = users_asking.sort_values('year_quarter', ascending=False)

    return users_asking


def get_users_last_90_days(df):
    """Get active users in last 90 days"""
    if df.empty:
        return pd.DataFrame()

    last_90_df = df[df['recent_90_days'] == 1]
    summary = last_90_df.groupby('user_name').agg({
        'daily_questions': 'sum',
        'question_date': ['nunique', 'max']
    }).reset_index()

    summary.columns = ['user_name', 'question_count', 'visit_count', 'most_recent_visit']
    summary = summary.sort_values(['visit_count', 'most_recent_visit'], ascending=[False, False])
    return summary.head(1000)


def create_90_day_rolling_averages(df):
    """Create 90-day rolling averages"""
    if df.empty:
        return pd.DataFrame()

    df['question_date'] = pd.to_datetime(df['question_date'])
    start_date = df['question_date'].min()
    end_date = df['question_date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date)

    rolling_stats = []
    for single_date in all_dates:
        start_window = single_date - pd.Timedelta(days=89)
        filtered_df = df[(df['question_date'] >= start_window) & (df['question_date'] <= single_date)]

        unique_users = filtered_df['user_name'].nunique()
        total_questions = filtered_df['daily_questions'].sum()

        rolling_stats.append({
            'date': single_date.strftime('%Y-%m-%d'),
            'unique_users': unique_users,
            'total_questions': total_questions
        })

    return pd.DataFrame(rolling_stats)


def create_7_day_rolling_averages(df):
    """Create 7-day rolling averages"""
    if df.empty:
        return pd.DataFrame()

    df['question_date'] = pd.to_datetime(df['question_date'])
    start_date = df['question_date'].min()
    end_date = df['question_date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date)

    rolling_stats = []
    for single_date in all_dates:
        start_window = single_date - pd.Timedelta(days=6)
        filtered_df = df[(df['question_date'] >= start_window) & (df['question_date'] <= single_date)]

        unique_users = filtered_df['user_name'].nunique()
        total_questions = filtered_df['daily_questions'].sum()

        rolling_stats.append({
            'date': single_date.strftime('%Y-%m-%d'),
            'unique_users': unique_users,
            'total_questions': total_questions
        })

    return pd.DataFrame(rolling_stats)


def render_layout(monthly_cohort, quarterly_cohort, rmau, rqau, active_users, rolling_90, rolling_7, tenants):
    """Render all visualizations using templates"""
    DEFAULT_HEIGHT = 50
    table_template = jinja2.Template(TABLE_TEMPLATE)
    chart_template = jinja2.Template(CHART_TEMPLATE)
    visualizations = []

    subtitle = f"Tenant: {', '.join(tenants)}; As of: {datetime.now().strftime('%Y-%m-%d')}"

    # 1. Monthly Cohort Table
    if not monthly_cohort.empty:
        monthly_cohort_reset = monthly_cohort.reset_index()
        # Convert any Period columns to strings
        for col in monthly_cohort_reset.columns:
            if pd.api.types.is_period_dtype(monthly_cohort_reset[col]):
                monthly_cohort_reset[col] = monthly_cohort_reset[col].astype(str)
        table_vars = {
            'dfs': [monthly_cohort_reset],
            "height": DEFAULT_HEIGHT,
            "subtitle": subtitle,
            "warnings": None,
            "title": "Monthly Cohort Analysis"
        }
        rendered = table_template.render(**table_vars)
        visualizations.append(SkillVisualization(title="Monthly Cohort", layout=rendered))

    # 2. Quarterly Cohort Table
    if not quarterly_cohort.empty:
        quarterly_cohort_reset = quarterly_cohort.reset_index()
        # Convert any Period columns to strings
        for col in quarterly_cohort_reset.columns:
            if pd.api.types.is_period_dtype(quarterly_cohort_reset[col]):
                quarterly_cohort_reset[col] = quarterly_cohort_reset[col].astype(str)
        table_vars = {
            'dfs': [quarterly_cohort_reset],
            "height": DEFAULT_HEIGHT,
            "subtitle": subtitle,
            "warnings": None,
            "title": "Quarterly Cohort Analysis"
        }
        rendered = table_template.render(**table_vars)
        visualizations.append(SkillVisualization(title="Quarterly Cohort", layout=rendered))

    # 3. RMAU Table
    if not rmau.empty:
        table_vars = {
            'dfs': [rmau],
            "height": DEFAULT_HEIGHT,
            "subtitle": subtitle,
            "warnings": None,
            "title": "Returning Monthly Active Users (RMAU)"
        }
        rendered = table_template.render(**table_vars)
        visualizations.append(SkillVisualization(title="RMAU", layout=rendered))

    # 4. RQAU Table
    if not rqau.empty:
        table_vars = {
            'dfs': [rqau],
            "height": DEFAULT_HEIGHT,
            "subtitle": subtitle,
            "warnings": None,
            "title": "Returning Quarterly Active Users (RQAU)"
        }
        rendered = table_template.render(**table_vars)
        visualizations.append(SkillVisualization(title="RQAU", layout=rendered))

    # 5. Active Users Table
    if not active_users.empty:
        table_vars = {
            'dfs': [active_users],
            "height": DEFAULT_HEIGHT,
            "title": "Active Users (Last 90 Days)",
            "subtitle": subtitle,
            "warnings": None
        }
        rendered = table_template.render(**table_vars)
        visualizations.append(SkillVisualization(title="Active Users", layout=rendered))

    # 6. 90-Day Rolling Chart
    if not rolling_90.empty:
        chart_vars = {
            "subtitle": subtitle,
            "chart_title": "90 Day Rolling Active Users and Total Questions",
            "dates": rolling_90["date"].tolist(),
            "unique_users": rolling_90["unique_users"].tolist(),
            "total_questions": rolling_90["total_questions"].tolist(),
            "height": DEFAULT_HEIGHT,
            "warnings": None
        }
        rendered = chart_template.render(**chart_vars)
        visualizations.append(SkillVisualization(title="90 Day Rolling Totals", layout=rendered))

    # 7. 7-Day Rolling Chart
    if not rolling_7.empty:
        chart_vars = {
            "subtitle": subtitle,
            "chart_title": "7 Day Rolling Active Users and Total Questions",
            "dates": rolling_7["date"].tolist(),
            "unique_users": rolling_7["unique_users"].tolist(),
            "total_questions": rolling_7["total_questions"].tolist(),
            "height": DEFAULT_HEIGHT,
            "warnings": None
        }
        rendered = chart_template.render(**chart_vars)
        visualizations.append(SkillVisualization(title="7 Day Rolling Totals", layout=rendered))

    return visualizations   


if __name__ == '__main__':
    skill_input: SkillInput = scorecard.create_input(
        arguments={'tenants': ['cpw prod']}
    )
    out = scorecard(skill_input)
    preview_skill(scorecard, out)
