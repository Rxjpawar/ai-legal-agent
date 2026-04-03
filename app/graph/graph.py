from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.constants import START, END

from app.agents.research_agent import research_agent
from app.agents.analysis_agent import analysis_agent
from app.agents.drafting_agent import drafting_agent
from app.agents.review_agent import review_agent

class State(TypedDict):
    query: str
    retrieved_context: str | None
    analysis: str | None
    draft_document: str | None
    reviewed_document: str | None

def build_graph():

    graph_builder = StateGraph(State)

    graph_builder.add_node("research_agent", research_agent)
    graph_builder.add_node("analysis_agent", analysis_agent)
    graph_builder.add_node("drafting_agent", drafting_agent)
    graph_builder.add_node("review_agent", review_agent)

    graph_builder.add_edge(START, "research_agent")
    graph_builder.add_edge("research_agent", "analysis_agent")
    graph_builder.add_edge("analysis_agent", "drafting_agent")
    graph_builder.add_edge("drafting_agent", "review_agent")
    graph_builder.add_edge("review_agent", END)

    return graph_builder.compile()