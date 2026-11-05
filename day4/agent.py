# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from langchain.agents import create_agent
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

model = init_chat_model(
    model="openai:gemini-3.1-flash-lite",
    temperature=0,
    # api_key=getenv("GEMINI_API_KEY"),
)


agent = create_agent(
    model=model,
    tools=[create_ticket],
)

result = agent.invoke({
    "messages": messages
})


print(result['messages'][-1].text)

# What is agent?
#   - An agent is a system where a language model decides what to do next, takes actions through tools, observes the results, and repeats until the task is done.