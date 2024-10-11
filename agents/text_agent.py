from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console

load_dotenv()

CONSOLE = Console()


class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str
    progress: str


LLM = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)


def update_progress(state: State, message: str) -> State:
    """Update the progress in the state and print it"""
    CONSOLE.print(f"[bold blue]{message}[/bold blue]")
    state["progress"] = message
    return state


def classification_node(state: Annotated[State, "NodeState"]):
    """Classify the text and update progress"""
    state = update_progress(state, "Classifying text...")
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText:{text}\n\nCategory:",
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = LLM.invoke([message]).content.strip()
    state["classification"] = classification
    return update_progress(state, "Classification complete.")


def entity_extraction_node(state: Annotated[State, "NodeState"]):
    """Extract entities and update progress"""
    state = update_progress(state, "Extracting entities...")
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:",
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = LLM.invoke([message]).content.strip().split(", ")
    state["entities"] = entities
    return update_progress(state, "Entity extraction complete.")


def summarization_node(state: Annotated[State, "NodeState"]):
    """Summarize the text and update progress"""
    state = update_progress(state, "Summarizing text...")
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:",
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = LLM.invoke([message]).content.strip()
    state["summary"] = summary
    return update_progress(state, "Summarization complete.")


def create_graph():
    workflow = StateGraph(State)

    # Add nodes to the graph
    workflow.add_node("classification_node", classification_node)
    workflow.add_node("entity_extraction", entity_extraction_node)
    workflow.add_node("summarization", summarization_node)

    # Set up the workflow
    workflow.set_entry_point("classification_node")
    workflow.add_edge("classification_node", "entity_extraction")
    workflow.add_edge("entity_extraction", "summarization")
    workflow.add_edge("summarization", END)

    # Compile the graph
    app = workflow.compile()

    return app


def test():
    sample_text = """
    OpenAI has announced the GPT-4 model, which is a large multimodal model that exhibits human-level performance on various professional benchmarks. It is developed to improve the alignment and safety of AI systems.
    Additionally, the model is designed to be more efficient and scalable than its predecessor, GPT-3. The GPT-4 model is expected to be released in the coming months and will be available to the public for research and development purposes.
    """

    app = create_graph()
    state_input = {"text": sample_text, "progress": "Starting workflow..."}

    CONSOLE.print(Panel.fit("Starting AI Agent Workflow", style="bold magenta"))
    result = app.invoke(state_input)

    CONSOLE.print(Panel.fit("AI Agent Workflow Results", style="bold green"))
    rprint("[bold cyan]Classification:[/bold cyan]", result["classification"])
    rprint("\n[bold cyan]Entities:[/bold cyan]", ", ".join(result["entities"]))
    rprint("\n[bold cyan]Summary:[/bold cyan]", result["summary"])
