from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

load_dotenv()

CONSOLE = Console()


class PlannerState(TypedDict):
    messages: Annotated[List[any], "The messages in the conversation"]
    city: str
    interests: List[str]
    itinerary: str


LLM = ChatOpenAI(model="gpt-4o-mini")

itinerary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary.",
        ),
        ("human", "Create an itinerary for my day trip."),
    ]
)


def input_city(state: PlannerState) -> PlannerState:
    city = Prompt.ask(
        "[bold cyan]Please enter the city you want to visit for your day trip[/bold cyan]"
    )
    return {
        **state,
        "city": city,
        "messages": state["messages"] + [HumanMessage(content=city)],
    }


def input_interests(state: PlannerState) -> PlannerState:
    interests_input = Prompt.ask(
        f"[bold cyan]Please enter your interests for the trip to {state['city']} (comma-separated)[/bold cyan]"
    )
    interests = [interest.strip() for interest in interests_input.split(",")]
    return {
        **state,
        "interests": interests,
        "messages": state["messages"] + [HumanMessage(content=interests_input)],
    }


def create_itinerary(state: PlannerState) -> PlannerState:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(
            description=f"[cyan]Creating an itinerary for {state['city']} based on your interests...[/cyan]",
            total=None,
        )
        response = LLM.invoke(
            itinerary_prompt.format_messages(
                city=state["city"], interests=", ".join(state["interests"])
            )
        )

    CONSOLE.print("\n[bold green]Final Itinerary:[/bold green]")
    CONSOLE.print(Panel(Markdown(response.content), expand=False, border_style="green"))

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "itinerary": response.content,
    }


def create_graph():
    workflow = StateGraph(PlannerState)

    workflow.add_node("input_city", input_city)
    workflow.add_node("input_interests", input_interests)
    workflow.add_node("create_itinerary", create_itinerary)

    workflow.set_entry_point("input_city")

    workflow.add_edge("input_city", "input_interests")
    workflow.add_edge("input_interests", "create_itinerary")
    workflow.add_edge("create_itinerary", END)

    app = workflow.compile()
    return app


def run_travel_planner(user_request: str):
    CONSOLE.print(
        Panel.fit(
            f"[bold magenta]Welcome to the Travel Planner![/bold magenta]\n\n[italic]{user_request}[/italic]",
            border_style="magenta",
        )
    )

    state = {
        "messages": [HumanMessage(content=user_request)],
        "city": "",
        "interests": [],
        "itinerary": "",
    }

    app = create_graph()

    for output in app.stream(state):
        pass  # The nodes themselves now handle all printing

    CONSOLE.print(
        Panel.fit(
            "[bold magenta]Thank you for using the Travel Planner![/bold magenta]",
            border_style="magenta",
        )
    )


def test():
    user_request = "I want to plan a day trip."
    run_travel_planner(user_request)
