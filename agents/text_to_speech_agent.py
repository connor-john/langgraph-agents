from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv
import io
import tempfile
import re
import os
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

load_dotenv()

CLIENT = OpenAI()
CONSOLE = Console()


class AgentState(TypedDict):
    input_text: str
    processed_text: str
    audio_data: bytes
    audio_path: str
    content_type: str


def sanitize_filename(text, max_length=20):
    """Convert text to a valid and concise filename."""
    sanitized = re.sub(r"[^\w\s-]", "", text.lower())
    sanitized = re.sub(r"[-\s]+", "_", sanitized)
    return sanitized[:max_length]


def classify_content(state: AgentState) -> AgentState:
    """Classify the input text into one of four categories: general, poem, news, or joke."""
    response = CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Classify the content as one of: 'general', 'poem', 'news', 'joke'.",
            },
            {"role": "user", "content": state["input_text"]},
        ],
    )
    state["content_type"] = response.choices[0].message.content.strip().lower()
    return state


def process_general(state: AgentState) -> AgentState:
    """Process general content (no specific processing, return as-is)."""
    state["processed_text"] = state["input_text"]
    return state


def process_poem(state: AgentState) -> AgentState:
    """Process the input text as a poem, rewriting it in a poetic style."""
    response = CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Rewrite the following text as a short, beautiful poem:",
            },
            {"role": "user", "content": state["input_text"]},
        ],
    )
    state["processed_text"] = response.choices[0].message.content.strip()
    return state


def process_news(state: AgentState) -> AgentState:
    """Process the input text as news, rewriting it in a formal news anchor style."""
    response = CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Rewrite the following text in a formal news anchor style:",
            },
            {"role": "user", "content": state["input_text"]},
        ],
    )
    state["processed_text"] = response.choices[0].message.content.strip()
    return state


def process_joke(state: AgentState) -> AgentState:
    """Process the input text as a joke, turning it into a short, funny joke."""
    response = CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Turn the following text into a short, funny joke:",
            },
            {"role": "user", "content": state["input_text"]},
        ],
    )
    state["processed_text"] = response.choices[0].message.content.strip()
    return state


def text_to_speech(state: AgentState, save_file: bool = False) -> AgentState:
    """
    Converts processed text into speech using a voice mapped to the content type.
    Optionally saves the audio to a file.

    Args:
        state (AgentState): Dictionary containing the processed text and content type.
        save_file (bool, optional): If True, saves the audio to a file. Defaults to False.

    Returns:
        AgentState: Updated state with audio data and file path (if saved).
    """

    # Map content type to a voice, defaulting to "alloy"
    voice_map = {"general": "alloy", "poem": "nova", "news": "onyx", "joke": "shimmer"}
    voice = voice_map.get(state["content_type"], "alloy")

    audio_data = io.BytesIO()

    # Generate speech and stream audio data into memory
    with CLIENT.audio.speech.with_streaming_response.create(
        model="tts-1", voice=voice, input=state["processed_text"]
    ) as response:
        for chunk in response.iter_bytes():
            audio_data.write(chunk)

    state["audio_data"] = audio_data.getvalue()

    # Save audio to a file if requested
    if save_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(state["audio_data"])
            state["audio_path"] = temp_audio.name
    else:
        state["audio_path"] = ""

    return state


def create_graph():
    # Define the graph
    workflow = StateGraph(AgentState)

    # Add nodes to the graph
    workflow.add_node("classify_content", classify_content)
    workflow.add_node("process_general", process_general)
    workflow.add_node("process_poem", process_poem)
    workflow.add_node("process_news", process_news)
    workflow.add_node("process_joke", process_joke)
    workflow.add_node("text_to_speech", text_to_speech)

    # Set the entry point of the graph
    workflow.set_entry_point("classify_content")

    # Define conditional edges based on content type
    workflow.add_conditional_edges(
        "classify_content",
        lambda x: x["content_type"],
        {
            "general": "process_general",
            "poem": "process_poem",
            "news": "process_news",
            "joke": "process_joke",
        },
    )

    # Connect processors to text-to-speech
    workflow.add_edge("process_general", "text_to_speech")
    workflow.add_edge("process_poem", "text_to_speech")
    workflow.add_edge("process_news", "text_to_speech")
    workflow.add_edge("process_joke", "text_to_speech")

    # Compile the graph
    app = workflow.compile()
    return app


def run_tts_agent_and_play(input_text: str, content_type: str, save_file: bool = True):
    app = create_graph()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Processing input...", total=None)
        result = app.invoke(
            {
                "input_text": input_text,
                "processed_text": "",
                "audio_data": b"",
                "audio_path": "",
                "content_type": content_type,
            }
        )
        progress.update(task, completed=True)

    CONSOLE.print(
        Panel(
            f"[bold green]Detected content type:[/bold green] [yellow]{result['content_type']}[/yellow]"
        )
    )

    CONSOLE.print("\n[bold green]Processed text:[/bold green]")
    CONSOLE.print(Panel(result["processed_text"], expand=False, border_style="green"))

    if save_file:
        audio_dir = "./data"
        os.makedirs(audio_dir, exist_ok=True)

        sanitized_text = sanitize_filename(input_text)
        file_name = f"{content_type}_{sanitized_text}.mp3"
        file_path = os.path.join(audio_dir, file_name)

        with open(file_path, "wb") as f:
            f.write(result["audio_data"])

        CONSOLE.print(
            Panel(
                f"[bold green]Audio saved to:[/bold green] [blue]{file_path}[/blue]",
                expand=False,
            )
        )
    else:
        CONSOLE.print("[yellow]Audio not saved to file.[/yellow]")

    return result


def test():
    examples = {
        "general": "The quick brown fox jumps over the lazy dog.",
        "poem": "Roses are red, violets are blue, AI is amazing, and so are you!",
        "news": "Breaking news: Scientists discover a new species of deep-sea creature in the Mariana Trench.",
        "joke": "Why don't scientists trust atoms? Because they make up everything!",
    }

    table = Table(title="TTS Agent Test Results")
    table.add_column("Content Type", style="cyan")
    table.add_column("Input Text", style="magenta")
    table.add_column("Processed Text", style="green")
    table.add_column("Audio File", style="yellow")

    for content_type, text in examples.items():
        CONSOLE.print(
            f"\n[bold cyan]Processing example for {content_type} content:[/bold cyan]"
        )
        CONSOLE.print(Panel(f"[bold]Input text:[/bold] {text}", expand=False))

        result = run_tts_agent_and_play(text, content_type, save_file=True)

        table.add_row(
            content_type,
            text[:30] + "..." if len(text) > 30 else text,
            result["processed_text"][:30] + "..."
            if len(result["processed_text"]) > 30
            else result["processed_text"],
            os.path.basename(result["audio_path"])
            if result["audio_path"]
            else "Not saved",
        )

        CONSOLE.print("=" * 80)

    CONSOLE.print("\n[bold green]Summary of all processed examples:[/bold green]")
    CONSOLE.print(table)

    CONSOLE.print(
        "\n[bold green]All examples processed. You can find the audio files in the 'data' directory.[/bold green]"
    )
