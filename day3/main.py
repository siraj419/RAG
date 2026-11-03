import json
from enum import Enum
from logging import CRITICAL
import os
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

SYSTEM_PROMPT = """You are TicketBot, an assistant that creates IT support tickets.

Always use the create_ticket tool to create tickets. Never say a ticket was created without calling it.

Priority rules: CRITICAL for outages, security issues, or data loss affecting all users. HIGH for major broken features, deadlines, or many users affected. MEDIUM for real issues with a workaround or limited impact. LOW for minor bugs, cosmetic issues, or feature requests.

If priority is ambiguous, ask one clarifying question before creating the ticket. If the core issue itself is unclear, ask before creating the ticket. Do not invent details, error codes, or assignees that were not stated.

Routing when no assignee is given: login, authentication, SSO, or password issues go to Identity. Server, outage, downtime, or infrastructure issues go to Infrastructure. Billing, invoice, or payment issues go to Finance Ops. Otherwise leave assignee unset.

Keep responses short and professional. After creating a ticket, confirm the ticket ID, priority, and assignee in one sentence. Do not repeat the full description back to the user."""


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CreateTicketArgs(BaseModel):
    title: str = Field(description="Short summary of the issue")
    description: str = Field(description="Detailed description of the problem")
    priority: Priority = Field(description="Urgency level of the ticket")
    assignee: Optional[str] = Field(default=None, description="Person or team to assign the ticket to")


tools = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a support ticket in the system",
            "parameters": CreateTicketArgs.model_json_schema()
        }
    }
]


def create_ticket(title, description, priority, assignee=None):
    ticket = {
        "id": 42,
        "title": title,
        "description": description,
        "priority": priority,
        "assignee": assignee,
        "status": "open"
    }
    return json.dumps(ticket)


messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Our production server is down, nobody can log in. Assign this to Bob, it's urgent."}
]

response = client.chat.completions.create(
    model="gemini-3.1-flash-lite",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

msg = response.choices[0].message
messages.append(msg)

print(msg)

if msg.tool_calls:
    for tool_call in msg.tool_calls:
        if tool_call.function.name == "create_ticket":
            raw_args = json.loads(tool_call.function.arguments)
            validated_args = CreateTicketArgs(**raw_args)

            result = create_ticket(
                title=validated_args.title,
                description=validated_args.description,
                priority=validated_args.priority.value,
                assignee=validated_args.assignee
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    final = client.chat.completions.create(model="gemini-3.1-flash-lite", messages=messages)
    print(final.choices[0].message.content)

# ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_9176', function=Function(arguments='{"title":"Production server outage","priority":"critical","assignee":"Infrastructure","description":"Production server is down, preventing all user logins."}', name='create_ticket'), type='function', extra_content={'google': {'thought_signature': 'EnMKcQFpFH0TbFZKMrFsCkPVW+7+0eBcyzuFn7j6cM/DbZ2+Oe5elfz/Q1o8udaifsQ+zPw2tuVKROzY2YSWVH/ij6HNmqUhJysrBWlJaMWEWtoiCowwtI8C578hX8at6Jnk5ESziw3OO6CCZzneJHMVTzKv'}})])
# Ticket 42 has been created with CRITICAL priority and assigned to Infrastructure.

