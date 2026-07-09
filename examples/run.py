"""
Final run for Open Coscientist with streaming output.

This drives the idea-generation pipeline using the IAU404 technosignatures
prompt (`technosignatures_iaus404.md`). The prompt's `{{CONTEXT_FILE}}`
placeholder is substituted with the absolute path to the IAU404 corpus
(`00_COMBINED_PLAYLIST.md`).

Prerequisites:
- Academic MCP server running (see ACADEMIC_MCP_SERVER_URL in .env)
- Root .env filled in (OPENROUTER_API_KEY, MODEL_NAME, etc.)

Speed knobs (all optional, set in .env):
  INITIAL_HYPOTHESES_COUNT   default 4  (2 tool-based + 2 debate)
  MAX_ITERATIONS             default 1  (one evolve loop)
  EVOLUTION_MAX_COUNT        default 2
  COSCIENTIST_DEV_MODE=true  → fewer lit-review papers (still full pipeline)
"""
import os
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

# Fast full-feature defaults: every node stays on, counts stay scientifically useful.
INITIAL_HYPOTHESES_COUNT = int(os.getenv("INITIAL_HYPOTHESES_COUNT", "4"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "1"))
EVOLUTION_MAX_COUNT = int(os.getenv("EVOLUTION_MAX_COUNT", "2"))
ENABLE_LITERATURE_REVIEW = os.getenv("ENABLE_LITERATURE_REVIEW_NODE", "true").lower() in (
    "true",
    "1",
    "yes",
)
ENABLE_TOOL_CALLING = os.getenv("ENABLE_TOOL_CALLING_GENERATION", "true").lower() in (
    "true",
    "1",
    "yes",
)

# IAU404 technosignatures prompt + meeting corpus.
# Override with env vars so another machine can point at local copies:
#   export IAU404_PROMPT_FILE=/path/to/technosignatures_iaus404.md
#   export IAU404_CONTEXT_FILE=/path/to/00_COMBINED_PLAYLIST.md
_DEFAULT_PROMPTS_DIR = Path(
    "/home/andromedalactea/universityDevelops/thesis/coai/coai-denario/develop-eggs/prompts"
)
PROMPT_FILE = Path(
    os.getenv("IAU404_PROMPT_FILE", str(_DEFAULT_PROMPTS_DIR / "technosignatures_iaus404.md"))
)
CONTEXT_FILE = Path(
    os.getenv("IAU404_CONTEXT_FILE", str(_DEFAULT_PROMPTS_DIR / "00_COMBINED_PLAYLIST.md"))
)


def load_goal() -> str:
    if not PROMPT_FILE.is_file():
        raise FileNotFoundError(
            f"IAU404 prompt not found: {PROMPT_FILE}\n"
            "Set IAU404_PROMPT_FILE to the path of technosignatures_iaus404.md"
        )
    if not CONTEXT_FILE.is_file():
        raise FileNotFoundError(
            f"IAU404 meeting corpus not found: {CONTEXT_FILE}\n"
            "Set IAU404_CONTEXT_FILE to the path of 00_COMBINED_PLAYLIST.md"
        )
    text = PROMPT_FILE.read_text(encoding="utf-8")
    return text.replace("{{CONTEXT_FILE}}", str(CONTEXT_FILE.resolve()))


goal = load_goal()


async def main():
    console = Console()
    tools_count = max(1, INITIAL_HYPOTHESES_COUNT // 2) if ENABLE_TOOL_CALLING else 0
    debate_count = INITIAL_HYPOTHESES_COUNT - tools_count if ENABLE_TOOL_CALLING else INITIAL_HYPOTHESES_COUNT
    if ENABLE_TOOL_CALLING and debate_count == 0:
        tools_count = INITIAL_HYPOTHESES_COUNT
        debate_count = 0

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
            f"Academic MCP: [bold]{os.getenv('ACADEMIC_MCP_SERVER_URL', 'http://localhost:8889/mcp')}[/bold]\n"
            f"Hypotheses: [bold]{INITIAL_HYPOTHESES_COUNT}[/bold] "
            f"({tools_count} tool-based + {debate_count} debate) | "
            f"evolve top [bold]{EVOLUTION_MAX_COUNT}[/bold] × [bold]{MAX_ITERATIONS}[/bold] iter\n"
            f"Features: lit_review=[bold]{ENABLE_LITERATURE_REVIEW}[/bold]  "
            f"tool_gen=[bold]{ENABLE_TOOL_CALLING}[/bold]  "
            f"dev_mode=[bold]{os.getenv('COSCIENTIST_DEV_MODE', 'false')}[/bold]  "
            f"reasoning=[bold]{os.getenv('DEEPSEEK_REASONING_EFFORT', '(default)')}[/bold]",
            title="[cyan]Config (fast full-feature)[/cyan]",
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
        max_iterations=MAX_ITERATIONS,
        initial_hypotheses_count=INITIAL_HYPOTHESES_COUNT,
        evolution_max_count=EVOLUTION_MAX_COUNT,
        tools_config="src/open_coscientist/config/examples/academic_semantic_scholar.yaml",
    )

    reporter = ConsoleReporter()

    last_state = await reporter.run(
        event_stream=generator.generate_hypotheses(
            research_goal=research_goal,
            progress_callback=default_progress_callback,
            opts={
                "enable_literature_review_node": ENABLE_LITERATURE_REVIEW,
                "enable_tool_calling_generation": ENABLE_TOOL_CALLING,
            },
            stream=True,
        ),
        research_goal=research_goal,
    )


if __name__ == "__main__":
    # wrap with run_console for graceful shutdown on KeyboardInterrupt and hide internal warnings
    run_console(main())
