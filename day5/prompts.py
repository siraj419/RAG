from unittest import result

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the following question: {question}"
)

prompt = prompt.invoke({"question": "What is the capital of France?"})

print(prompt)  

# messages=[HumanMessage(content='You are a helpful assistant. Answer the following question: What is the capital of France?', additional_kwargs={}, response_metadata={})]

msg_temp = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the following question: {question}"
    ),
    (
        "human",
        "{question}"
    )
])

# message = mmsg_temp.invoke({"question": "What is the capital of France?"})
# print(message)

# messages=[SystemMessage(content='You are a helpful assistant. Answer the following question: What is the capital of France?', additional_kwargs={}, response_metadata={}), HumanMessage(content='What is the capital of France?', additional_kwargs={}, response_metadata={})]

model = init_chat_model(
    model="google_genai:gemini-3.1-flash-lite",
    temperature=0,
)

# chain
chain = msg_temp | model | StrOutputParser()

# reult = chain.invoke({"question": "What is the capital of France?"}).text

print(chain.invoke({"question": "What is the capital of France?"}))