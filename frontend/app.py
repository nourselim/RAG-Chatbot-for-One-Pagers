import streamlit as st
import time
from datetime import datetime
from pathlib import Path
import sys
import os
import json
from typing import List, Dict, Any, Optional, Tuple

# Add the rag folder to the path so we can import from it
rag_path = Path(__file__).parent.parent / "rag"
sys.path.append(str(rag_path))

from faiss_service import FaissCandidateSearch
from embed_only import OUT_DIR, EMB_NPY, META_JSONL

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="DeBotte AI - Employee Skills Finder", 
    page_icon="", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize LangChain components
llm = ChatOpenAI(
    model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Pydantic models for structured output
class QueryIntent(BaseModel):
    intent: str = Field(description="One of: greeting, search, compare, followup_detail, clarify, out_of_scope")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    reasoning: str = Field(description="Brief explanation of the intent classification")
    needs_clarification: bool = Field(description="Whether the query needs clarification")
    clarification_question: str = Field(description="Question to ask if clarification needed, empty string if not")
    is_out_of_scope: bool = Field(description="Whether the query is outside DeBotte's allowed scope")
    scope_explanation: str = Field(description="Explanation of why the query is out of scope, empty string if in scope")

class EmployeeReference(BaseModel):
    names: List[str] = Field(description="List of employee names referenced in the query")
    confidence: float = Field(description="Confidence in name extraction 0.0-1.0")
    source: str = Field(description="How the names were identified: explicit, pronoun, ordinal, or inferred")

class SearchQuery(BaseModel):
    query: str = Field(description="Rewritten search query for employee search")
    target_names: List[str] = Field(description="Specific employee names to search for")
    chunk_types: List[str] = Field(description="Types of information to focus on: skills, experience, certifications, etc.")
    top_k: int = Field(description="Number of results requested, default 5", default=5)

class SearchResult(BaseModel):
    employee_name: str = Field(description="Full name of the employee")
    title: str = Field(description="Job title")
    email: str = Field(description="Email address")
    relevance_summary: str = Field(description="One-line summary of why this employee matches the query")
    confidence: float = Field(description="Match confidence score 0.0-1.0")
    match_type: str = Field(description="Type of match: exact, partial, or related")

class ResultAnalysis(BaseModel):
    has_exact_matches: bool = Field(description="Whether there are exact matches for the query")
    exact_count: int = Field(description="Number of exact matches")
    partial_count: int = Field(description="Number of partial/related matches")
    summary_message: str = Field(description="User-friendly message explaining the results")
    should_show_partial: bool = Field(description="Whether to show partial matches")

# LangChain chains and prompts
intent_prompt = ChatPromptTemplate.from_template("""
You are DeBotte, an AI assistant for a Deloitte employee skills finder system. Your scope is LIMITED to helping find employees based on their skills, experience, certifications, and work history.

ALLOWED QUERIES:
- Finding employees with specific skills (e.g., "AWS", "Python", "SAP")
- Finding employees with specific experience (e.g., "DevOps", "Project Management")
- Finding employees with certifications (e.g., "AWS Certified", "PMP")
- Finding employees who worked with specific clients or industries
- Comparing employees' skills and experience
- Getting detailed information about specific employees
- General greetings and help requests

OUT OF SCOPE QUERIES:
- Personal information (smoking, health, personal habits)
- Non-work related questions
- Questions about company policies, salaries, or confidential information
- Questions about other companies or competitors
- Questions unrelated to employee skills and work experience

Classify the user's intent into one of these categories:
- greeting: Salutations, hello, hi, etc.
- search: Looking for employees with specific skills/experience
- compare: Comparing two or more employees
- followup_detail: Asking for more details about a previously mentioned employee
- clarify: Query is unclear, incomplete, or needs more context
- out_of_scope: Query is outside DeBotte's allowed scope

User query: {query}
Recent context: {context}

Return a JSON object with:
- intent: the classified intent
- confidence: confidence score 0.0-1.0
- reasoning: brief explanation
- needs_clarification: true if query needs clarification
- clarification_question: specific question to ask if clarification needed
- is_out_of_scope: true if query is outside allowed scope
- scope_explanation: explanation of why it's out of scope, empty string if in scope
""")

intent_chain = LLMChain(
    llm=llm,
    prompt=intent_prompt,
    output_parser=JsonOutputParser(pydantic_object=QueryIntent)
)

name_extraction_prompt = ChatPromptTemplate.from_template("""
Extract employee names from the user's query and recent conversation context.

User query: {query}
Recent candidates: {recent_candidates}
All available employees: {all_employees}

Return a JSON object with:
- names: list of employee names found
- confidence: confidence score 0.0-1.0
- source: how names were identified (explicit, pronoun, ordinal, or inferred)
""")

name_extraction_chain = LLMChain(
    llm=llm,
    prompt=name_extraction_prompt,
    output_parser=JsonOutputParser(pydantic_object=EmployeeReference)
)

query_rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the user's query into a clear, searchable format for finding employees.

Original query: {query}
Context: {context}

Return a JSON object with:
- query: rewritten search query
- target_names: specific employee names to search for
- chunk_types: types of information to focus on
- top_k: number of results requested
""")

query_rewrite_chain = LLMChain(
    llm=llm,
    prompt=query_rewrite_prompt,
    output_parser=JsonOutputParser(pydantic_object=SearchQuery)
)

result_analysis_prompt = ChatPromptTemplate.from_template("""
Analyze search results to determine if there are exact matches or only partial/related matches.

User query: {query}
Search results: {results}

For each result, determine if it's an EXACT match or PARTIAL/RELATED:

EXACT MATCH criteria:
- Direct mention of the requested skill/technology in the text
- Certification in the requested area
- Explicit experience with the requested technology
- Text directly answers the user's question

PARTIAL/RELATED criteria:
- Indirectly related skills or experience
- Similar technologies or domains
- General background that might be relevant
- No direct mention of the requested skill

Examples:
- Query: "Kubernetes experience" 
  - EXACT: "Kubernetes Certified Administrator (CKA)", "deployed with Kubernetes"
  - PARTIAL: "DevOps experience", "containerization knowledge"

Return a JSON object with:
- has_exact_matches: true if any results are exact matches
- exact_count: number of exact matches (must be accurate)
- partial_count: number of partial/related matches (must be accurate)
- summary_message: user-friendly message explaining the results
- should_show_partial: whether to show partial matches to the user

IMPORTANT: exact_count + partial_count must equal the total number of results provided.
""")

result_analysis_chain = LLMChain(
    llm=llm,
    prompt=result_analysis_prompt,
    output_parser=JsonOutputParser(pydantic_object=ResultAnalysis)
)

out_of_scope_prompt = ChatPromptTemplate.from_template("""
You are DeBotte, an AI assistant for a Deloitte employee skills finder. The user has asked something outside your allowed scope.

Explain politely and professionally that you cannot help with this request, and redirect them to what you CAN help with.

User query: {query}
Scope explanation: {scope_explanation}

Write a friendly, professional response that:
1. Acknowledges their question
2. Explains why it's outside your scope
3. Redirects them to what you can help with
4. Keeps it under 50 words

Response:""")

out_of_scope_chain = LLMChain(
    llm=llm,
    prompt=out_of_scope_prompt
)

# Initialize FAISS service
@st.cache_resource(show_spinner=False)
def load_faiss_service():
    """Load the FAISS service for candidate search."""
    try:
        if not (OUT_DIR / "faiss_index.bin").exists():
            st.warning("FAISS index not found. Please run the RAG pipeline first.")
            st.info("Run this command in the rag/ folder: python main.py auto")
            return None
        
        service = FaissCandidateSearch(OUT_DIR)
        service.load_index()
        return service
    except Exception as e:
        st.error(f"Error loading FAISS service: {str(e)}")
        return None

faiss_service = load_faiss_service()

# Helper functions
def get_recent_candidates(messages: List[Dict]) -> List[str]:
    """Get names from the most recent candidate list."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("candidates"):
            return [c.get("name", "") for c in msg["candidates"] if c.get("name")]
    return []

def get_all_employee_names() -> List[str]:
    """Get all available employee names from FAISS metadata."""
    if not faiss_service or not hasattr(faiss_service, "metas"):
        return []
    
    names = []
    seen = set()
    for meta in faiss_service.metas:
        name = meta.get("employee_name", "").strip()
        if name and name.lower() not in seen:
            names.append(name)
            seen.add(name.lower())
    return names

def analyze_search_results(query: str, results: List[Tuple]) -> ResultAnalysis:
    """Use LangChain to analyze search results and determine match quality."""
    try:
        # Prepare results for analysis
        result_summaries = []
        for emp_id, (score, cos_score, meta) in results:
            name = meta.get("employee_name", "Unknown")
            text = meta.get("text", "")[:200]
            chunk_type = meta.get("chunk_type", "")
            result_summaries.append(f"{name}: {chunk_type} - {text}")
        
        result_text = "\n".join(result_summaries)
        
        # Analyze with LangChain using improved prompt
        analysis_result = result_analysis_chain.run({
            "query": query,
            "results": result_text
        })
        
        # Parse result
        if isinstance(analysis_result, str):
            analysis_data = json.loads(analysis_result)
        else:
            analysis_data = analysis_result
        
        # Validate the analysis results
        total_results = len(results)
        exact_count = analysis_data.get("exact_count", 0)
        partial_count = analysis_data.get("partial_count", 0)
        
        # Ensure counts are consistent
        if exact_count + partial_count != total_results:
            # Fallback: use semantic analysis to determine exact vs partial
            exact_count = 0
            partial_count = 0
            
            for emp_id, (score, cos_score, meta) in results:
                text = meta.get("text", "").lower()
                query_lower = query.lower()
                
                # Check for exact matches
                if any(term in text for term in query_lower.split()):
                    exact_count += 1
                else:
                    partial_count += 1
        
        return ResultAnalysis(
            has_exact_matches=exact_count > 0,
            exact_count=exact_count,
            partial_count=partial_count,
            summary_message=analysis_data.get("summary_message", f"Found {total_results} employees"),
            should_show_partial=partial_count > 0
        )
        
    except Exception as e:
        st.error(f"Result analysis error: {str(e)}")
        # Fallback analysis
        total_results = len(results)
        return ResultAnalysis(
            has_exact_matches=False,
            exact_count=0,
            partial_count=total_results,
            summary_message=f"Found {total_results} potentially relevant employees.",
            should_show_partial=True
        )

def generate_relevance_summary(employee_name: str, query: str, chunks: List[Dict]) -> str:
    """Generate a relevance summary using LangChain."""
    if not chunks:
        return f"Limited information available for {employee_name}"
    
    summary_prompt = ChatPromptTemplate.from_template("""
    Given the user's query and employee information, write a one-line summary explaining why this employee is relevant.
    Keep it under 20 words and focus on the specific skills/experience mentioned in the query.
    
    Query: {query}
    Employee: {employee_name}
    Information: {chunks}
    
    Summary:""")
    
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    
    try:
        chunk_text = "\n".join([c.get("text", "")[:200] for c in chunks[:3]])
        result = summary_chain.run({
            "query": query,
            "employee_name": employee_name,
            "chunks": chunk_text
        })
        return result.strip()
    except Exception:
        return f"Relevant experience available for {employee_name}"

def handle_greeting(query: str) -> str:
    """Handle greeting queries."""
    greeting_prompt = ChatPromptTemplate.from_template("""
    You are a friendly AI assistant called DeBotte for an employee skills finder. Respond to the user's greeting in their language.
    In one sentence, say you can help find employees at Deloitte by skills, experience, or certifications.
    Keep it under 25 words.
    
    User greeting: {query}
    """)
    
    greeting_chain = LLMChain(llm=llm, prompt=greeting_prompt)
    try:
        return greeting_chain.run({"query": query}).strip()
    except Exception:
        return "Hello! I'm DeBotte, your Deloitte Skills Finder. Ask me about employee skills, experience, or certifications."

def handle_search(query: str, search_params: SearchQuery) -> Tuple[str, List[Dict]]:
    """Handle search queries with proper error handling and result analysis."""
    try:
        # Perform search with guardrails
        if not faiss_service:
            return "Search service is not available. Please run the RAG pipeline first.", []
        
        # Extract target names if specified
        target_names = search_params.target_names if search_params.target_names else None
        
        # Perform search
        results = faiss_service.search_filtered(
            query=search_params.query,
            target_names=target_names,
            chunk_types=search_params.chunk_types,
            top_k=50,
            pool_size=search_params.top_k
        )
        
        if not results:
            return "I couldn't find any employees matching your query. Try being more specific or ask about different skills.", []
        
        # Analyze results using LangChain
        analysis = analyze_search_results(query, results)
        
        # Format response based on analysis
        if analysis.has_exact_matches:
            if analysis.partial_count > 0:
                response = f"Found {analysis.exact_count} employees with exact matches for your query and {analysis.partial_count} with related experience:\n\n"
            else:
                response = f"Found {analysis.exact_count} employees with exact matches for your query:\n\n"
        else:
            response = f"No exact matches found for your query, but here are {analysis.partial_count} employees with related experience:\n\n"
        
        candidates = []
        
        for i, (emp_id, (score, cos_score, meta)) in enumerate(results, 1):
            name = meta.get("employee_name", "Unknown")
            title = meta.get("title", "")
            email = meta.get("email", "")
            
            # Get additional chunks for summary
            chunks = faiss_service.search_employee_details(name, query=search_params.query, top_k=3)
            summary = generate_relevance_summary(name, search_params.query, chunks)
            
            response += f"**{i}. {name}**"
            if title:
                response += f" - {title}"
            if email:
                response += f" ({email})"
            response += f"\n{summary}\n\n"
            
            candidates.append({
                'name': name,
                'title': title,
                'email': email,
                'text': summary,
                'confidence': '',
                'score': cos_score
            })
        
        return response, candidates
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return "I encountered an error while searching. Please try rephrasing your query.", []

def handle_compare(query: str, names: List[str]) -> Tuple[str, List[Dict]]:
    """Handle comparison queries."""
    if len(names) < 2:
        return "Please specify at least two employees to compare.", []
    
    try:
        # Get details for each employee
        all_chunks = {}
        candidates = []
        
        for name in names[:2]:  # Limit to 2 for comparison
            chunks = faiss_service.search_employee_details(name, query=query, top_k=5)
            all_chunks[name] = chunks
            
            if chunks:
                candidates.append({
                    'name': name,
                    'title': chunks[0].get('title', ''),
                    'email': chunks[0].get('email', ''),
                    'text': f"Comparison data available for {name}",
                    'confidence': '',
                    'score': 0.0
                })
        
        if not all_chunks:
            return "I couldn't find enough information to make a comparison.", []
        
        # Generate comparison using LangChain
        comparison_prompt = ChatPromptTemplate.from_template("""
        Compare two employees based on the user's query. Write a concise, helpful comparison.
        Focus on the specific aspects mentioned in the query.
        
        Query: {query}
        Employee 1 ({name1}): {chunks1}
        Employee 2 ({name2}): {chunks2}
        
        Comparison:""")
        
        comparison_chain = LLMChain(llm=llm, prompt=comparison_prompt)
        
        result = comparison_chain.run({
            "query": query,
            "name1": names[0],
            "name2": names[1],
            "chunks1": "\n".join([c.get("text", "")[:200] for c in all_chunks.get(names[0], [])]),
            "chunks2": "\n".join([c.get("text", "")[:200] for c in all_chunks.get(names[1], [])])
        })
        
        return f"**Comparison of {names[0]} and {names[1]}:**\n\n{result.strip()}", candidates
        
    except Exception as e:
        st.error(f"Comparison error: {str(e)}")
        return "I encountered an error while comparing employees. Please try again.", []

def handle_followup(query: str, names: List[str]) -> Tuple[str, List[Dict]]:
    """Handle follow-up detail queries."""
    if not names:
        return "I'm not sure which employee you're referring to. Could you please specify a name?", []
    
    name = names[0]  # Focus on first mentioned name
    
    try:
        # Get detailed information
        chunks = faiss_service.search_employee_details(name, query=query, top_k=8)
        
        if not chunks:
            return f"I couldn't find additional details for {name}. Try asking about their skills, certifications, or experience.", []
        
        # Generate narrative using LangChain
        narrative_prompt = ChatPromptTemplate.from_template("""
        Write a friendly, concise paragraph about the employee based on the provided information.
        Answer the user's specific question using only the facts provided.
        Keep it under 100 words.
        
        Question: {query}
        Employee: {name}
        Information: {chunks}
        
        Answer:""")
        
        narrative_chain = LLMChain(llm=llm, prompt=narrative_prompt)
        
        chunk_text = "\n".join([c.get("text", "")[:200] for c in chunks[:5]])
        result = narrative_chain.run({
            "query": query,
            "name": name,
            "chunks": chunk_text
        })
        
        # Create candidate for display
        candidates = [{
            'name': name,
            'title': chunks[0].get('title', ''),
            'email': chunks[0].get('email', ''),
            'text': result.strip(),
            'confidence': '',
            'score': 0.0
        }]
        
        return result.strip(), candidates
        
    except Exception as e:
        st.error(f"Follow-up error: {str(e)}")
        return f"I encountered an error while getting details for {name}. Please try again.", []

# CSS Styling
st.markdown("""
<style>
    .main { padding: 0; }
    .chat-container { max-width: 800px; margin: 0 auto; padding: 20px; }
    .user-message { background-color: #40414f; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; margin-left: 50px; margin-right: 0; }
    .assistant-message { background-color: #444654; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; margin-right: 50px; margin-left: 0; }
    .stTextInput > div > div > input { background-color: #40414f; color: white; border: 1px solid #565869; border-radius: 10px; padding: 15px; font-size: 16px; }
    .css-1d391kg { background-color: black; }
    .stButton > button { background-color: #26890d; color: white; border: none; border-radius: 5px; padding: 10px 20px; font-size: 14px; cursor: pointer; }
    section[data-testid="stSidebar"] { background-color: #000000 !important; }
    section[data-testid="stSidebar"] .css-1d391kg { background-color: #000000 !important; }
    section[data-testid="stSidebar"] * { color: white !important; }
    .stButton > button:hover { background-color: #1f6d0b; }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #2d2d2d; }
    ::-webkit-scrollbar-thumb { background: #888; border-radius: 4px; }
    .chat-history { background-color: #202123; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .model-selector { background-color: #202123; color: white; border: 1px solid #565869; border-radius: 5px; padding: 5px; }
</style>
""", unsafe_allow_html=True)

# State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session" not in st.session_state:
    st.session_state.current_session = "default"

# Sidebar
with st.sidebar:
    st.title("DeBotte AI")
    st.markdown("**Employee Skills Finder**")
    st.markdown(
        "<div style='height:2px; background:#ffffff; margin:0.75rem 0;'></div>",
        unsafe_allow_html=True
    )
    
    # New Chat Button
    if st.button("New Chat", use_container_width=True):
        new_session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.chat_sessions[new_session_id] = []
        st.session_state.current_session = new_session_id
        st.session_state.messages = []
        st.rerun()
    
    st.markdown(
        "<div style='height:2px; background:#ffffff; margin:0.75rem 0;'></div>",
        unsafe_allow_html=True
    )
    
    # Recent Chats
    st.subheader("Recent Chats")
    for session_id in list(st.session_state.chat_sessions.keys()):
        if st.button(f"Chat {session_id[-6:]}", key=session_id, use_container_width=True):
            st.session_state.current_session = session_id
            st.session_state.messages = st.session_state.chat_sessions[session_id]
            st.rerun()
    
    # Clear All Chats
    if st.button("Clear All Chats", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_sessions = {}
        st.session_state.current_session = "default"
        st.rerun()
    
    st.markdown(
        "<div style='height:2px; background:#ffffff; margin:0.75rem 0;'></div>",
        unsafe_allow_html=True
    )
    
    st.markdown("**Instructions:**")
    st.markdown(
        """
        1. Ask about employee skills, experience, or certifications
        2. Use natural language queries
        3. Example: "Find employees with AWS certification"
        4. Example: "Who has experience with SAP?"
        """
    )

# Main Chat Interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about employee skills, experience, or certifications..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Step 1: Classify intent
            recent_context = get_recent_candidates(st.session_state.messages)
            intent_result = intent_chain.run({
                "query": prompt,
                "context": ", ".join(recent_context) if recent_context else "No recent context"
            })
            
            # Parse intent result
            try:
                intent_data = json.loads(intent_result) if isinstance(intent_result, str) else intent_result
                intent = intent_data.get("intent", "search")
                needs_clarification = intent_data.get("needs_clarification", False)
                clarification_question = intent_data.get("clarification_question", "")
                is_out_of_scope = intent_data.get("is_out_of_scope", False)
                scope_explanation = intent_data.get("scope_explanation", "")
            except:
                intent = "search"
                needs_clarification = False
                clarification_question = ""
                is_out_of_scope = False
                scope_explanation = ""
            
            # Handle out-of-scope queries
            if is_out_of_scope:
                response = out_of_scope_chain.run({
                    "query": prompt,
                    "scope_explanation": scope_explanation
                })
                candidates = []
            # Handle clarification needed
            elif needs_clarification:
                response = clarification_question or "Could you please clarify your question?"
                candidates = []
            else:
                # Step 2: Extract names if needed
                names = []
                if intent in ["compare", "followup_detail"]:
                    name_result = name_extraction_chain.run({
                        "query": prompt,
                        "recent_candidates": ", ".join(recent_context),
                        "all_employees": ", ".join(get_all_employee_names())
                    })
                    
                    try:
                        name_data = json.loads(name_result) if isinstance(name_result, str) else name_result
                        names = name_data.get("names", [])
                    except:
                        names = []
                
                # Step 3: Handle based on intent
                if intent == "greeting":
                    response = handle_greeting(prompt)
                    candidates = []
                
                elif intent == "search":
                    # Rewrite query for search
                    query_result = query_rewrite_chain.run({
                        "query": prompt,
                        "context": ", ".join(recent_context) if recent_context else "No recent context"
                    })
                    
                    try:
                        search_params = json.loads(query_result) if isinstance(query_result, str) else query_result
                        search_query = search_params.get("query", prompt)
                        target_names = search_params.get("target_names", [])
                        chunk_types = search_params.get("chunk_types", [])
                        top_k = search_params.get("top_k", 5)
                        
                        # Create SearchQuery object
                        search_query_obj = SearchQuery(
                            query=search_query,
                            target_names=target_names,
                            chunk_types=chunk_types,
                            top_k=top_k
                        )
                        
                        response, candidates = handle_search(prompt, search_query_obj)
                        
                    except Exception as e:
                        st.error(f"Query rewrite error: {str(e)}")
                        response = "I encountered an error processing your query. Please try rephrasing it."
                        candidates = []
                
                elif intent == "compare":
                    response, candidates = handle_compare(prompt, names)
                
                elif intent == "followup_detail":
                    response, candidates = handle_followup(prompt, names)
                
                else:
                    response = "I'm not sure how to help with that. Could you please rephrase your question?"
                    candidates = []
            
            # Display response
            message_placeholder.markdown(response)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            response = "I encountered an unexpected error. Please try again."
            candidates = []
            message_placeholder.markdown(response)

    # Add assistant message to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "candidates": candidates
    })
    
    # Update session storage
    if st.session_state.current_session not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[st.session_state.current_session] = []
    st.session_state.chat_sessions[st.session_state.current_session] = st.session_state.messages.copy()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 12px;'>
       DeBotte AI - Employee Skills Finder | Built by Innov8 with Love
    </div>
    """,
    unsafe_allow_html=True
)