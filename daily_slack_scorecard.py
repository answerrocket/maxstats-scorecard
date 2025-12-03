from skill_framework import skill, SkillParameter, SkillInput, SkillOutput
from datetime import datetime
from answer_rocket import AnswerRocketClient
import json
import subprocess
import time

DEFAULT_TENANTS = ['abibg prod', 'cpw prod', 'suntorymax prod']
@skill(
    name="daily_slack_scorecard",
    description="Run scorecard for tenants and send results to Slack",
    parameters=[
        SkillParameter(
            name="tenants",
            is_multi=True,
            constrained_to="enums",
            description="tenants to analyze",
            constrained_values=['abibg prod', 'cpw prod', 'suntorymax prod']
        ),
        SkillParameter(
            name="slack_webhook_url",
            parameter_type="code",
            description="slack webhook url",
        )
    ]
)

def daily_slack_scorecard(parameters: SkillInput) -> SkillOutput:
    print('Running run_scorecard')
    tenants = parameters.arguments.tenants if hasattr(parameters.arguments, 'tenants') else DEFAULT_TENANTS
    slack_webhook_url = parameters.arguments.slack_webhook_url if hasattr(parameters.arguments, 'slack_webhook_url') else None
    print(f"slack_webhook_url: {slack_webhook_url}")
    print(f"tenants: {tenants}")
    agent_id = parameters.assistant_id
    share_links = run_questions(tenants, agent_id)
    time.sleep(60) # wait for report to be generated
    for tenant in share_links:
        share_link = share_links[tenant]
        message = f"Updated {tenant} Max product scorecard: {share_link}"
        send_slack_message(message, slack_webhook_url)

    return SkillOutput(narrative="Scorecard reports have been generated and sent to Slack.")
        
def run_questions(tenants, agent_id) -> dict[str, str]:
    print(f'received tenants: {tenants}, agent_id: {agent_id}')
    # Get the current date
    current_date = datetime.now()
    date_str = current_date.strftime('%m-%d-%Y')

    # Initialize AnswerRocket client
    arc = AnswerRocketClient()

    share_links = {}
    # Loop through tenants and send questions
    for tenant in tenants:
        thread = arc.chat.create_new_thread(copilot_id=agent_id)
        question = f"Today is {date_str}. Run Max Report for tenant {tenant}"
        response = arc.chat.queue_chat_question(thread_id=thread.id, question=question, skip_cache=True)
        thread_id = response.thread_id
        share_object = arc.chat.share_chat_thread(original_thread_id=thread_id)
        share_links[tenant] = share_object.link_to_shared_thread

    # Run report on all tenants
    if len(tenants) > 1:
        thread = arc.chat.create_new_thread(copilot_id=agent_id)
        question = f"Today is {date_str}. Run Max Report for all tenants: {', '.join(tenants)}. Analyze all of them into one report. Do not run multi-skills"
        response = arc.chat.queue_chat_question(thread_id=thread.id, question=question, skip_cache=True)
        thread_id = response.thread_id
        share_object = arc.chat.share_chat_thread(original_thread_id=thread_id)
        share_links['All'] = share_object.link_to_shared_thread

    return share_links

def send_slack_message(message: str, slack_webhook_url: str):
    if not slack_webhook_url:
        raise ValueError("Slack webhook URL is required")

    """Send a message to Slack via webhook."""
    payload = json.dumps({"text": message})
    print(f"Sending message: {payload}")  # for debugging


    process = subprocess.run(
        ["curl", "-X", "POST", "-H", "Content-type: application/json", "--data", payload, slack_webhook_url],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return process.stdout, process.stderr