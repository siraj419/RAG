from os import getenv
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

texts = [
    "Assignments must be submitted on the student portal before 11:59 PM on the due date.",
    "Late submissions lose 10 percent of the marks for each day.",
    "An extension can be requested from the course instructor at least 48 hours.",
    "Classes run from 9 AM to 4 PM, Monday to Friday. Lunch break is from 1 PM to 2 PM.",
    "The library is open from 8 AM to 10 PM on weekdays and 10 AM to 4 PM on weekends.",
    "Students may borrow up to 5 books for 14 days. Late returns are fined 50 rupees per day.",
    "Final exam timetables are posted on the notice board and portal two weeks before exams begin.",
    "Students must maintain 75 percent attendance to be allowed to sit the final exam.",
    "Fee payment is due by the 10th of each month at the accounts office.",
]

docs = [Document(page_content=t) for t in texts]



client = QdrantClient(url="http://localhost:6333")

if not client.collection_exists("retrieval"):
    client.create_collection(
        collection_name="retrieval",
        vectors_config={"dense": {"size": len(embeddings.embed_query("probe")), "distance": "Cosine"}},
        sparse_vectors_config={"sparse": {"modifier": "idf"}}
    )

store = QdrantVectorStore(
    client=client,
    collection_name="retrieval",
    embedding=embeddings,
    sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse"
)

# store.add_documents(docs)

# for d in store.similarity_search("I missed my deadline, what happens?", k=5):
#     print(d.page_content)


# Query Rewriting
#   - Reformulate the query to be more specific or clear.
# HyDe
#   - Hypothetical Document.
# ReRanking
#   - Re-rank the retrieved documents based on relevance to the query.

# from pydantic import BaseModel

class Queries(BaseModel):
    queries: list[str]

expand_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You support literature and course-material search at a university.\n"
     "Given a student's question, write 3 distinct scholarly search queries that together cover it.\n"
     "- Vary the terminology: use synonyms, technical terms, and closely related concepts.\n"
     "- If the question has several parts, write one query per part.\n"
     "- Do not introduce concepts that are unrelated to the question."),
    ("human", "{question}"),
])
expand_chain = expand_prompt | llm.with_structured_output(Queries)

# q = "How do I know my results aren't just luck, and how do I avoid a biased sample?"
# queries = [q] + expand_chain.invoke({"question": q}).queries     # keep the original
# print(queries)

# seen, combined = set(), []
# for query in queries:
#     for d in store.similarity_search(query, k=3):
#         if d.page_content not in seen:
#             seen.add(d.page_content)
#             combined.append(d)

# for d in combined:
#     print(d.page_content[:80])

hyde_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a university lecturer writing course notes.\n"
     "Write a brief passage (2 to 3 sentences) in the formal register of an academic textbook that "
     "would answer the student's question. Use precise technical terminology.\n"
     "The passage is used only as a search aid, so factual precision is less important than "
     "matching the style and vocabulary of scholarly course notes."),
    ("human", "{question}"),
])
hyde_chain = hyde_prompt | llm | StrOutputParser()

# q = "I missed my deadline, what happens?"
# passage = hyde_chain.invoke({"question": q})
# print(passage)

# for d in store.similarity_search(passage, k=2):
#     print(d.page_content)



from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question, candidates, top_n=3):
    scores = reranker.predict([(question, d.page_content) for d in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]

# q = "I missed my deadline, what happens?"
# candidates = store.similarity_search(q, k=5)   # fast and wide

# for d in candidates:
#     print(d.page_content)
    
# best = rerank(q, candidates, top_n=2)              # slow and precise

# for d in best:
#     print(d.page_content)

def retrieve(question, top_n=3):
    rewritten = expand_chain.invoke({"question": question})
    passage = hyde_chain.invoke({"question": question})

    seen, candidates = set(), []
    for query in [question, rewritten, passage]:
        for d in store.similarity_search(query, k=5):
            if d.page_content not in seen:
                seen.add(d.page_content)
                candidates.append(d)

    return rerank(question, candidates, top_n=top_n) 

answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a teaching assistant for a university course.\n"
     "Answer the student's question using only the course materials provided.\n"
     "- Write in clear, formal academic English suitable for a student.\n"
     "- Cite the source of each claim in square brackets, for example [ml_notes_ch3].\n"
     "- If the materials do not contain the answer, state that the answer is not covered "
     "and suggest consulting the instructor or the textbook. Do not speculate.\n"
     "- Keep the answer under 120 words."),
    ("human", "Course materials:\n{context}\n\nStudent question: {question}"),
])
answer_chain = answer_prompt | llm | StrOutputParser()

q = "Why does my model do great on training data but badly on new data?"
chunks = retrieve(q)
context = "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in chunks)
print(answer_chain.invoke({"context": context, "question": q}))