"""
Short end-to-end test of the idea-generation system with ALL features enabled.

Pipeline (Mode 3 - maximum literature grounding):
  supervisor -> literature_review -> generate (with live tool calls)
  -> reflection -> review -> ranking -> meta_review -> evolve
  -> review -> ranking -> proximity

Counts are kept small so the run finishes quickly while still exercising
every node. Credentials are read from the root .env file.

Prerequisites:
  - Academic MCP server running on http://localhost:8889
  - Root .env filled in (OPENROUTER_API_KEY, etc.)
"""
import os
import asyncio

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

GOAL = """Act as a computational astrobiologist. Propose original, testable
biosignature or technosignature hypotheses that could be validated with existing
public astronomical datasets and standard Python scientific libraries. Focus on
measurable spectral, photometric, or thermodynamic anomalies. Do not propose
building new software frameworks; propose scientific statements about nature."""


async def main():
    console = Console()
    console.print()
    console.print(Panel(GOAL, title="[cyan]Research Goal (short test)[/cyan]", border_style="cyan"))
    console.print(
        Panel(
            f"Pipeline model (all nodes): [bold]{MODEL_NAME}[/bold]\n"
            f"Academic MCP: [bold]{os.getenv('ACADEMIC_MCP_SERVER_URL', 'http://localhost:8889/mcp')}[/bold]\n"
            f"Features: literature_review + tool-calling generation + evolution",
            title="[cyan]Config[/cyan]",
            border_style="cyan",
        )
    )

    generator = HypothesisGenerator(
        model_name=MODEL_NAME,
        max_iterations=1,            # one refine/evolve loop (fast)
        initial_hypotheses_count=3,  # small for a quick test
        evolution_max_count=2,
        tools_config="src/open_coscientist/config/examples/academic_semantic_scholar.yaml",
    )

    reporter = ConsoleReporter()
    await reporter.run(
        event_stream=generator.generate_hypotheses(
            research_goal=GOAL,
            progress_callback=default_progress_callback,
            opts={
                "enable_literature_review_node": True,
                "enable_tool_calling_generation": True,
            },
            stream=True,
        ),
        research_goal=GOAL,
    )


if __name__ == "__main__":
    run_console(main())
