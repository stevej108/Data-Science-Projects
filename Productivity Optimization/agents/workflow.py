from langgraph.graph import StateGraph, END
from core.state import SearchState

from agents.discover import discover_documents
from agents.parser import parse_documents
from agents.retrieve import retrieve_documents
from agents.rerank_documents import rerank_documents
from agents.summarize import summarize_documents
from agents.synthesize import synthesize_documents
from agents.document_retrieval import retrieve_candidate_documents_node
from agents.memo import generate_memo


workflow = StateGraph(SearchState)

print("Registering discover")
workflow.add_node("discover", discover_documents)

print("Registering candidate_selection")
workflow.add_node("candidate_selection", retrieve_candidate_documents_node)

print("Registering parse")
workflow.add_node("parse", parse_documents)

print("Registering retrieve")
workflow.add_node("retrieve", retrieve_documents)

print("Registering rerank")
workflow.add_node("rerank", rerank_documents)

print("Registering summarize")
workflow.add_node("summarize", summarize_documents)

print("Registering synthesize")
workflow.add_node("synthesize", synthesize_documents)

print("Registering memo")
workflow.add_node("memo", generate_memo)

workflow.set_entry_point("discover")

workflow.add_edge("discover", "candidate_selection")
workflow.add_edge("candidate_selection", "parse")
workflow.add_edge("parse", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "summarize")
workflow.add_edge("summarize", "synthesize")
workflow.add_edge("synthesize", "memo")
workflow.add_edge("memo", END)

print('Compiling workflow...')
app = workflow.compile()
print('Workflow compiled successfully!')