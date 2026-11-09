import os
import requests
from os import getenv
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.embeddings import Embeddings


model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    google_api_key=getenv("GEMINI_API_KEY"),
)

# =====================================================
# TOOLS
# =====================================================
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "StudyPackDemo/1.0 (student@example.com)"}

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return the text of the best matching article."""
    hits = requests.get(API, headers=HEADERS, timeout=10, params={
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": 1, "format": "json",
    }).json()["query"]["search"]

    if not hits:
        return "NO_RESULTS"

    title = hits[0]["title"]
    pages = requests.get(API, headers=HEADERS, timeout=10, params={
        "action": "query", "prop": "extracts", "titles": title,
        "explaintext": 1, "format": "json",
    }).json()["query"]["pages"]

    text = next(iter(pages.values())).get("extract", "")
    return f"ARTICLE TITLE: {title}\n\n{text[:6000]}"   # cap size to protect the context window


@tool
def save_study_pack(filename: str, content: str) -> str:
    """Save markdown content to the study_packs folder and return the file path."""
    os.makedirs("study_packs", exist_ok=True)
    safe = "".join(c for c in filename if c.isalnum() or c in "-_") or "study_pack"
    path = f"study_packs/{safe}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# =====================================================
# STEP 1: request -> search query  (prompt template)
# =====================================================
query_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Turn the student's request into the best Wikipedia search query. "
     "Use 2 to 5 words. Reply with the query only, no quotes."),
    ("human", "{request}"),
])
query_chain = query_prompt | model | StrOutputParser()


# =====================================================
# STEP 2: call the tool  (plain function in the chain)
# =====================================================
def fetch(inputs: dict) -> dict:
    query = inputs["query"].strip()
    article = search_wikipedia.invoke({"query": query})
    if article == "NO_RESULTS":
        raise ValueError(f"No Wikipedia article found for: {query!r}")
    return {"query": query, "article": article, "level": inputs["level"]}


# =====================================================
# STEP 3: notes + quiz  (two more prompt templates)
# =====================================================
notes_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a tutor writing for {level} students.\n"
     "Use ONLY the article provided. Do not add outside facts.\n"
     "Write: a 2-sentence overview, then 5 to 7 bullet-point key ideas, "
     "then a short 'Key terms' list with one-line definitions."),
    ("human", "{article}"),
])
notes_chain = notes_prompt | model | StrOutputParser()


class Question(BaseModel):
    question: str
    options: list[str] = Field(description="Exactly 4 answer options")
    answer: str = Field(description="The correct option, copied exactly from options")
    explanation: str = Field(description="One sentence explaining why it is correct")

class Quiz(BaseModel):
    questions: list[Question]

quiz_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You write multiple-choice quizzes for {level} students.\n"
     "Write exactly 4 questions using ONLY facts from the article. "
     "Each question has 4 options and one correct answer."),
    ("human", "{article}"),
])
quiz_chain = quiz_prompt | model.with_structured_output(Quiz)


# =====================================================
# STEP 4: assemble + save  (calls the second tool)
# =====================================================
def assemble(d: dict) -> dict:
    lines = [f"# Study Pack: {d['query'].title()}", "", "## Notes", d["notes"], "",
             "## Practice Quiz", ""]
    for i, q in enumerate(d["quiz"].questions, 1):
        lines.append(f"**Q{i}. {q.question}**")
        for letter, opt in zip("ABCD", q.options):
            lines.append(f"- {letter}. {opt}")
        lines.append("")
    lines.append("## Answer Key")
    for i, q in enumerate(d["quiz"].questions, 1):
        lines.append(f"{i}. {q.answer}: {q.explanation}")

    markdown = "\n".join(lines)
    path = save_study_pack.invoke({
        "filename": d["query"].replace(" ", "_"),
        "content": markdown,
    })
    return {"path": path, "markdown": markdown}


# =====================================================
# THE FULL CHAIN
# =====================================================
study_chain = (
    RunnablePassthrough.assign(query=query_chain)                  # add "query"
    | RunnableLambda(fetch)                                        # tool call -> "article"
    | RunnablePassthrough.assign(notes=notes_chain, quiz=quiz_chain)  # 2 branches in parallel
    | RunnableLambda(assemble)                                     # build + save file
)

# study_chain.get_graph().print_ascii()

if __name__ == "__main__":
    result = study_chain.invoke({
        "request": "Explain how vaccines work for my biology class",
        "level": "high school",
    })
    print("Saved to:", result["path"])
    print(result["markdown"])
    
    # Workflow Vs Agent
    # wikipedia search in langchain