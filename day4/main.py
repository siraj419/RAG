from langchain_openai import ChatOpenAI


from dotenv import load_dotenv  
load_dotenv()

from os import getenv

print(getenv("GEMINI_API_KEY"))
model = ChatOpenAI(
    model_name="gemma-4-31b-it",
    temperature=0,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    api_key=getenv("GEMINI_API_KEY"),
)

result = model.invoke(
    "What is the capital of France?"
)

print(result)

# content='<thought>The user is asking for the capital of France.\nFrance.\nCapital city.\nParis.</thought>The capital of France is Paris.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 8, 'total_tokens': 35, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'gemma-4-31b-it', 'system_fingerprint': None, 'id': 'anXraor-Aved28oP2MTecQ', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a1a7c2-995b-73d4-bebe-b400f1f8eb1b-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 8, 'output_tokens': 7, 'total_tokens': 35, 'input_token_details': {}, 'output_token_details': {}}