from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from database import get_retriever
from dotenv import load_dotenv

load_dotenv()

class GraphState(TypedDict):
    question: str
    chat_history: List[dict] 
    documents: List[Document]
    answer: str
    web_search_needed: bool
    is_grounded: bool

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def contextualize_question(state: GraphState):
    question = state["question"]
    chat_history = state.get("chat_history", [])
    
    if not chat_history:
        return {"question": question}
        
    system = """Given a chat history and the latest user question, which might reference context 
    in the chat history, formulate a standalone question which can be understood without the chat history. 
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Chat History: \n{chat_history} \n\n Latest Question: {question}")
    ])
    
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    chain = prompt | llm
    standalone_question = chain.invoke({"chat_history": history_str, "question": question})
    
    return {"question": standalone_question.content}

def retrieve(state: GraphState):
    question = state["question"]
    retriever = get_retriever()
    docs = retriever.invoke(question)
    return {"documents": docs, "question": question}

def grade_documents(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    
    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
        
    structured_llm_grader = llm.with_structured_output(Grade)
    
    system = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keywords or semantic meaning related to the question, grade it as relevant. 
    Give a binary score 'yes' or 'no'."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    retrieval_grader = grade_prompt | structured_llm_grader
    
    relevant_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score.lower() == "yes":
            relevant_docs.append(d)
            
    web_search_needed = len(relevant_docs) == 0
    return {"documents": relevant_docs, "question": question, "web_search_needed": web_search_needed}

def web_search(state: GraphState):
    question = state["question"]
    documents = state.get("documents", [])
    
    tool = TavilySearchResults(max_results=3)
    docs = tool.invoke({"query": question})
    
    web_results = "\n\n".join([d["content"] for d in docs])
    web_document = Document(page_content=web_results, metadata={"source": "Tavily Web Search"})
    
    documents.append(web_document)
    return {"documents": documents, "question": question}

def generate_answer(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", [])
    
    context = "\n\n".join([f"Source ({d.metadata.get('source', 'Unknown')}):\n{d.page_content}" for d in documents])
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    
    system = """You are a technical assistant. Use the retrieved context to answer the question concisely. 
    If you don't know, say you don't know. Always cite the source at the end."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Chat History: \n{chat_history} \n\n Question: {question} \n\n Context: {context}")
    ])
    
    rag_chain = prompt | llm
    response = rag_chain.invoke({"chat_history": history_str, "question": question, "context": context})
    
    return {"answer": response.content}

def check_hallucination(state: GraphState):
    documents = state["documents"]
    answer = state["answer"]

    class GradeHallucination(BaseModel):
        binary_score: str = Field(description="Answer is grounded in facts, 'yes' or 'no'")

    structured_llm_grader = llm.with_structured_output(GradeHallucination)

    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is strictly based on and supported by the retrieved facts."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved Facts: \n\n {documents} \n\n Generated Answer: {answer}"),
    ])

    hallucination_grader = prompt | structured_llm_grader
    context = "\n\n".join([d.page_content for d in documents])
    
    score = hallucination_grader.invoke({"documents": context, "answer": answer})
    is_grounded = score.binary_score.lower() == "yes"

    return {"is_grounded": is_grounded}

def decide_to_generate(state: GraphState):
    if state.get("web_search_needed", False):
        return "web_search"
    return "generate"

def decide_after_hallucination_check(state: GraphState):
    is_grounded = state.get("is_grounded", True)
    if is_grounded:
        return "grounded"
    else:
        return "hallucinated"

workflow = StateGraph(GraphState)

workflow.add_node("contextualize_question", contextualize_question)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate_answer)
workflow.add_node("check_hallucination", check_hallucination)

workflow.set_entry_point("contextualize_question")
workflow.add_edge("contextualize_question", "retrieve")

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"web_search": "web_search", "generate": "generate"}
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "check_hallucination")
workflow.add_conditional_edges(
    "check_hallucination",
    decide_after_hallucination_check,
    {"grounded": END, "hallucinated": "web_search"}
)

rag_application = workflow.compile()