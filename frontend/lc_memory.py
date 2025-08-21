# frontend/lc_memory.py
from __future__ import annotations
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

DEFAULT_SYSTEM_PROMPT = """
You are DeBotte, an internal AI assistant at Deloitte designed to help managers quickly find the best-suited employees for projects using the latest One-Pager documents.
Your primary goal is to provide precise, professional, and conversational answers based solely on the retrieved content. You must never invent information or provide guesses beyond the documents.

GUIDELINES:
- Accuracy first: base factual answers on retrieved documents.
- Professional tone; concise; use bullet points for lists.
- Formatting: label Name, Skills, Experience, Clients clearly.
- Missing data: say what's missing. Never invent.

Formatting Rules for Employee Listings:
- ALWAYS present each employee in this exact format with proper spacing:
  • Name: [Full Name]
  • Title: [Job Title]
  • Skills: [Skill 1], [Skill 2], [Skill 3]
  • Relevant Experience: [Brief description]

- Use consistent bullet points (•) throughout
- Ensure proper spacing between each employee profile
- Never run multiple employee profiles together without separation

MEMORY POLICY:
- You MAY use chat_history to resolve pronouns and follow-ups (e.g., “they”, “this candidate”, “the previous list”).
- You MAY answer meta questions about the conversation itself (e.g., “what was my first question?”, “repeat my last message”)
  using chat_history even if documents don’t contain that information.
- For all employee facts (skills, experience, certifications, clients), still rely on the retrieved documents only.

Matching Logic:
- If an employee's job title or description exactly matches the requested role, present them as a perfect match
- If employees clearly have the required skills, present them as matches
- If no exact match exists, suggest the closest matches but be clear about the limitations

Answer clearly and directly.
""".strip()

def format_docs(docs, max_chars: int = 3000) -> str:
    """Turn retrieved LangChain Documents into a single context string."""
    parts, total = [], 0
    for d in docs or []:
        block = (d.page_content or "").strip()
        if not block:
            continue
        total += len(block) + 1
        if total > max_chars:
            break
        parts.append(block)
        parts.append("")  # blank line
    return "\n".join(parts).strip()

def make_chain_with_memory(retriever, system_prompt: str | None = None, model: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0)
    get_q = itemgetter("question")

    # ---- 1) Extract referenced names from history ----
    names_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "From chat_history and the latest user question, extract the FULL NAMES of the employees the user refers to. "
         "Use the most recent assistant answer to resolve pronouns like 'they'/'their'. "
         "Return ONLY a comma-separated list of names (no extra text). If none, return an empty string."),
        MessagesPlaceholder("chat_history"),
        ("user", "{question}")
    ]).partial(chat_history=[])
    names_chain = {"question": get_q, "chat_history": itemgetter("chat_history")} | names_prompt | llm | StrOutputParser()

    # ---- 2) Rewrite question into a stand-alone search query (history aware) ----
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the user's latest question into a concise, stand-alone search query including all entities and "
         "constraints from chat_history. Do NOT answer; only rewrite."),
        MessagesPlaceholder("chat_history"),
        ("user", "{question}")
    ]).partial(chat_history=[])
    rewrite_chain = {"question": get_q, "chat_history": itemgetter("chat_history")} | rewrite_prompt | llm | StrOutputParser()

    # ---- 3) Retrieval that is name-aware and filtered by metadata ----
    def search_with_names(inputs: dict):
        q = rewrite_chain.invoke(inputs)
        names_csv = names_chain.invoke(inputs).strip()
        target = {n.strip() for n in names_csv.split(",") if n.strip()}

        # Step A: broad retrieval
        docs = retriever.get_relevant_documents(q)

        # Step B: keep only docs whose metadata.name is in target (if we have names)
        if target:
            docs = [d for d in docs if d.metadata.get("name") in target]

            # Fallback: if nothing left, hit the vectorstore directly per name
            if not docs and hasattr(retriever, "vectorstore"):
                vs = retriever.vectorstore
                for n in target:
                    docs.extend(vs.similarity_search(f"{n} business skills key skills skills", k=5))
                # final filter by name again (some stores may return others)
                docs = [d for d in docs if d.metadata.get("name") in target]

        return docs
    
    context_chain = RunnableLambda(search_with_names) | RunnableLambda(format_docs)

    base = {
        "context": context_chain,
        "question": get_q,
    } | answer_prompt | llm

    # ---- 4) Compose final answering chain ----
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt or DEFAULT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Context from documents:\n{context}\n\nUser question:\n{question}\n\nAnswer:")
    ]).partial(chat_history=[])

    base = {
        "context": search_with_names | format_docs,
        "question": get_q,
    } | answer_prompt | llm

    # ---- 5) Memory wrapper ----
    _store: dict[str, InMemoryChatMessageHistory] = {}
    def get_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in _store:
            _store[session_id] = InMemoryChatMessageHistory()
        return _store[session_id]

    return RunnableWithMessageHistory(
        base,
        get_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    ) | StrOutputParser()