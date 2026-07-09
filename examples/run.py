"""
Final run for Open Coscientist with streaming output.

This drives the idea-generation pipeline using the IAU404 technosignatures
prompt (`technosignatures_iaus404.md`). The prompt's `{{CONTEXT_FILE}}`
placeholder is substituted with the absolute path to the IAU404 corpus
(`00_COMBINED_PLAYLIST.md`).

Prerequisites:
- Academic MCP server running (see ACADEMIC_MCP_SERVER_URL in .env)
- Root .env filled in (OPENROUTER_API_KEY, MODEL_NAME, etc.)
"""
import os
import asyncio
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv not installed; rely on already-exported environment variables
    pass

from open_coscientist import HypothesisGenerator
from open_coscientist.console import ConsoleReporter, default_progress_callback, run_console
from rich.console import Console
from rich.panel import Panel

MODEL_NAME = os.getenv("MODEL_NAME", "openrouter/deepseek/deepseek-v4-flash")

# IAU404 technosignatures prompt + corpus (intentionally ignoring astrobiology.md).
PROMPTS_DIR = Path(
    "/home/andromedalactea/universityDevelops/thesis/coai/coai-denario/develop-eggs/prompts"
)
PROMPT_FILE = PROMPTS_DIR / "technosignatures_iaus404.md"
CONTEXT_FILE = PROMPTS_DIR / "00_COMBINED_PLAYLIST.md"


def load_goal() -> str:
    text = PROMPT_FILE.read_text(encoding="utf-8")
    return text.replace("{{CONTEXT_FILE}}", str(CONTEXT_FILE))


goal = load_goal()


async def main():
    console = Console()
    console.print()
    console.print(
        Panel(
            goal,
            title="[cyan]Research Goal (IAU404 technosignatures)[/cyan]",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            f"Prompt file: [bold]{PROMPT_FILE}[/bold]\n"
            f"Context file: [bold]{CONTEXT_FILE}[/bold]\n"
            f"Pipeline model (all nodes): [bold]{MODEL_NAME}[/bold]\n"
            f"Academic MCP: [bold]{os.getenv('ACADEMIC_MCP_SERVER_URL', 'http://localhost:8889/mcp')}[/bold]",
            title="[cyan]Config[/cyan]",
            border_style="cyan",
        )
    )
    research_goal = console.input(
        "\n[bold cyan]Research goal (Enter to use the IAU404 prompt above):[/bold cyan] "
    ).strip()
    if not research_goal:
        research_goal = goal
        console.print("[dim]Using IAU404 technosignatures prompt as research goal.[/dim]")

    generator = HypothesisGenerator(
        model_name=MODEL_NAME,
        max_iterations=2,
        initial_hypotheses_count=7,
        evolution_max_count=4,
        tools_config="src/open_coscientist/config/examples/academic_semantic_scholar.yaml",
    )

    reporter = ConsoleReporter()

    last_state = await reporter.run(
        event_stream=generator.generate_hypotheses(
            research_goal=research_goal,
            progress_callback=default_progress_callback,
            opts={
                "enable_literature_review_node": True,
                "enable_tool_calling_generation": True,
            },
            stream=True,
        ),
        research_goal=research_goal,
    )


if __name__ == "__main__":
    # wrap with run_console for graceful shutdown on KeyboardInterrupt and hide internal warnings
    run_console(main())
