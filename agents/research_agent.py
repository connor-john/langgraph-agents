import logging
from typing import Dict, TypedDict
from langgraph.graph import Graph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define the state of our system
class AgentState(TypedDict):
    query: str
    search_results: str
    summary: str
    review: str
    final_output: str
    iteration: int
    operation: str


# Initialize tools
web_search = DuckDuckGoSearchRun()
arxiv_search = ArxivAPIWrapper()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")
logger.info("Initialized tools and LLM")


# Define node functions
def web_search_node(state: AgentState) -> Dict:
    logger.debug(f"Entering web_search_node with query: {state['query']}")
    results = web_search.run(state["query"])
    state["search_results"] = (
        state["search_results"] + "\n\nWeb Search Results:\n" + results
    )
    logger.debug(
        f"Web search results added. Current search_results length: {len(state['search_results'])}"
    )
    return state


def arxiv_search_node(state: AgentState) -> Dict:
    logger.debug(f"Entering arxiv_search_node with query: {state['query']}")
    results = arxiv_search.run(state["query"])
    state["search_results"] = (
        state["search_results"] + "\n\nArXiv Search Results:\n" + results
    )
    logger.debug(
        f"ArXiv search results added. Current search_results length: {len(state['search_results'])}"
    )
    return state


def summarize_node(state: AgentState) -> Dict:
    logger.debug("Entering summarize_node")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a research assistant. Summarize the following search results.",
            ),
            ("human", "{search_results}"),
        ]
    )
    summary = llm.invoke(prompt.format_messages(search_results=state["search_results"]))
    state["summary"] = summary.content
    logger.debug(f"Summary generated. Length: {len(state['summary'])}")
    return state


def review_node(state: AgentState) -> Dict:
    logger.debug("Entering review_node")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a research reviewer. Review the following summary and suggest improvements or areas for further research. Specifically say 'further research needed' if further research is needed. DO NOT say 'further research needed' if the result is good.",
            ),
            ("human", "{summary}"),
        ]
    )
    review = llm.invoke(prompt.format_messages(summary=state["summary"]))
    state["review"] = review.content
    logger.debug(f"Review generated. Length: {len(state['review'])}")
    return state


def decide_continue(state: AgentState) -> str:
    logger.debug(f"Entering decide_continue. Current iteration: {state['iteration']}")
    if "further research needed" in state["review"].lower() and state["iteration"] < 3:
        state["operation"] = "continue"
        logger.debug("Decided to continue research")
    else:
        state["operation"] = "finish"
        logger.debug("Decided to finish research")
    return state


def update_query(state: AgentState) -> Dict:
    logger.debug("Entering update_query")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the review, generate a new, more focused search query.",
            ),
            ("human", "Original query: {query}\nReview: {review}"),
        ]
    )
    new_query = llm.invoke(
        prompt.format_messages(query=state["query"], review=state["review"])
    )
    state["query"] = new_query.content
    state["iteration"] += 1
    logger.debug(
        f"Query updated. New query: {state['query']}, New iteration: {state['iteration']}"
    )
    return state


def finish_node(state: AgentState) -> Dict:
    logger.debug("Entering finish_node")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate a final comprehensive research summary based on all the information gathered.",
            ),
            ("human", "Query: {query}\nSummary: {summary}\nReview: {review}"),
        ]
    )
    final_output = llm.invoke(prompt.format_messages(**state))
    state["final_output"] = final_output.content
    logger.debug(f"Final output generated. Length: {len(state['final_output'])}")
    return state


def create_graph():
    logger.info("Creating graph")
    workflow = Graph()

    workflow.add_node("web_search", web_search_node)
    workflow.add_node("arxiv_search", arxiv_search_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("review", review_node)
    workflow.add_node("update_query", update_query)
    workflow.add_node("finish", finish_node)
    workflow.add_node("decide_continue", decide_continue)

    workflow.add_edge("web_search", "arxiv_search")
    workflow.add_edge("arxiv_search", "summarize")
    workflow.add_edge("summarize", "review")
    workflow.add_edge("review", "decide_continue")

    workflow.set_entry_point("web_search")

    workflow.add_conditional_edges(
        "decide_continue",
        lambda x: x["operation"],
        {
            "continue": "update_query",
            "finish": "finish",
        },
    )
    workflow.add_edge("update_query", "web_search")
    workflow.add_edge("finish", END)

    app = workflow.compile()
    logger.info("Graph created and compiled")
    return app


def run_research(query: str) -> AgentState:
    logger.info(f"Starting research with query: {query}")
    app = create_graph()
    initial_state = {
        "query": query,
        "search_results": "",
        "summary": "",
        "review": "",
        "final_output": "",
        "iteration": 0,
        "operation": "continue",
    }
    logger.debug(f"Initial state: {initial_state}")
    result = app.invoke(initial_state)
    logger.info("Research completed")
    logger.debug(f"Final result: {result}")
    return result


def test():
    logger.info("Starting test")
    result = run_research("Recent advancements in AI agents")
    if result:
        logger.info(
            f"Test completed. Final output: {result.get('final_output', 'No final output generated')}"
        )
    else:
        logger.error("Test failed. Result is None")
    return result
