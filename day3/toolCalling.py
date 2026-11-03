import json
from dotenv import load_dotenv
import os
from openai import OpenAI


load_dotenv()

# Tool:
#   - Function is a function that a model can use to perform a specific task that it can't itself.





client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]

# 2. The real function
def get_weather(city, unit="celsius"):
    return json.dumps({"city": city, "temperature": 40, "unit": unit, "condition": "Sunny"})

# 3. Start conversation
messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

response = client.chat.completions.create(
    model="gemma-4-31b-it",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

msg = response.choices[0].message
messages.append(msg)

print(msg)

# 4. If the model wants to call a tool, run it and loop back
if msg.tool_calls:
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        if call.function.name == "get_weather":
            result = get_weather(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result
            })

    final = client.chat.completions.create(model="gemma-4-31b-it", messages=messages, tools=tools)
    print(final.choices[0].message.content)
else:
    print(msg.content)
    

# ChatCompletionMessage(content='<thought>The user is asking about the weather in Tokyo. 
# I should look for a tool that can provide weather information. The `get_weather` tool takes a `city` parameter and an optional `unit` parameter. In this caseThe user is asking about the weather in Tokyo. I should look for a tool that can provide weather information. The `get_weather` tool takes a `city` parameter and an optional `unit` parameter. In this case, the city is "Tokyo". I will call the `get_weather` tool with `city=\'Tokyo\'`.</thought>', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_1006147', function=Function(arguments='{"city":"Tokyo"}', name='get_weather'), type='function', extra_content={'google': {'thought_signature': 'EiYKJGUyNDgzMGE3LTVjZDYtNDJmZS05OThiLWVlNTM5ZTcyYjljMw=='}})], extra_content={'google': {'thought': True}})
# The current weather in Tokyo is sunny with a temperature of 40°C.