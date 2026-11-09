from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)



# result = model.embed_query("Hello world")
result = model.embed_documents(["Hello world", "Hello again", "random text"])


print(cosine_similarity(result[0], result[1]))
print(cosine_similarity(result[0], result[2]))