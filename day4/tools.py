from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()


from os import getenv

import json

@tool
def create_ticket(
    title: str,
    description: str,
    priority: str,
    assignee: str | None = None
) -> str:
    """
        Create a support ticket in the system.
        
        Args:
            title (str): The title of the ticket.
            description (str): A detailed description of the issue.
            priority (str): The priority level of the ticket (e.g., "low", "medium", "high").
            assignee (str, optional): The name of the person to whom the ticket is assigned. Defaults to None.
    """
    ticket = {
        "id": 42,
        "title": title,
        "description": description,
        "priority": priority,
        "assignee": assignee,
        "status": "open"
    }
    return json.dumps(ticket)

system_message = SystemMessage(content="""You are TicketBot, an assistant that creates IT support tickets.

Always use the create_ticket tool to create tickets. Never say a ticket was created without calling it.

Priority rules: CRITICAL for outages, security issues, or data loss affecting all users. HIGH for major broken features, deadlines, or many users affected. MEDIUM for real issues with a workaround or limited impact. LOW for minor bugs, cosmetic issues, or feature requests.

If priority is ambiguous, ask one clarifying question before creating the ticket. If the core issue itself is unclear, ask before creating the ticket. Do not invent details, error codes, or assignees that were not stated.

Routing when no assignee is given: login, authentication, SSO, or password issues go to Identity. Server, outage, downtime, or infrastructure issues go to Infrastructure. Billing, invoice, or payment issues go to Finance Ops. Otherwise leave assignee unset.

Keep responses short and professional. After creating a ticket, confirm the ticket ID, priority, and assignee in one sentence. Do not repeat the full description back to the user.
Once the create_ticket tool has returned a result, do not call it again. Reply with the one-sentence confirmation.
""")

user_message = HumanMessage(content="Our production server is down, nobody can log in. Assign this to Bob, it's urgent.")


messages = [system_message, user_message]

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    api_key=getenv("GEMINI_API_KEY"),
)

model_with_tools = model.bind_tools([create_ticket])
resp = model_with_tools.invoke(messages)
messages.append(resp)
print(resp)

if resp.tool_calls:
    if resp.tool_calls[0]['name'] == "create_ticket":
        args = resp.tool_calls[0]['args']
        result = create_ticket.invoke(args)
        messages.append(ToolMessage(content=result, tool_name="create_ticket", tool_call_id=resp.tool_calls[0]['id']))

final_resp = model_with_tools.invoke(messages)
print(final_resp.text)


# content=[] additional_kwargs={'function_call': {'name': 'create_ticket', 'arguments': '{"assignee": "Bob", "description": "Production server is down and users are unable to log in.", "title": "Production server outage", "priority": "CRITICAL"}'}, '__gemini_function_call_thought_signatures__': {'call_9002987172610638785': 'EnMKcQFpFH0Tb7dPe83AQ7gCbJC7i7bivaxC+thghWbEE4MVQA0rqBRRMIDvti03d9zROeI0flhfjRveT5TqCIU2AVNhnTuL9j8Tz2WDw/vAjzgjfCuB9Rpt3kVSwCbngsdlNdJDGXN2pGbDT3LjB4tVtTLK'}} response_metadata={'finish_reason':'STOP', 'model_name': 'gemini-3.1-flash-lite', 'safety_ratings': [], 'model_provider': 'google_genai'} id='lc_run--01a1acbd-e0e7-75a2-92d7-6f37f112d067-0' tool_calls=[{'name': 'create_ticket', 'args': {'assignee': 'Bob', 'description': 'Production server is down and users are unable to log in.', 'title': 'Production server outage', 'priority': 'CRITICAL'}, 'id': 'call_9002987172610638785', 'type': 'tool_call'}] invalid_tool_calls=[] usage_metadata={'input_tokens': 445, 'output_tokens': 49, 'total_tokens': 494, 'input_token_details': {'cache_read': 0}}
# Ticket 42 has been created with CRITICAL priority and assigned to Bob.