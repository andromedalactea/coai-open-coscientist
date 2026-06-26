"""
Example for Open Coscientist with streaming output.

This demonstrates hypothesis generation with literature review integration,
showing real-time streaming of results as they're generated.
"""
from open_coscientist import HypothesisGenerator
from open_coscientist.console import ConsoleReporter, default_progress_callback, run_console
# install rich in your environment
from rich.console import Console
from rich.panel import Panel
"""
Prerequisites:
- MCP server running (on http://localhost:8888/mcp)
- Set OPEN_AI_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY in your environment before running,
which depends on the MODEL_NAME you set below.
"""

MODEL_NAME = "openai/gpt-5-nano"

goal = """
Role:
Act as an autonomous computational scientist specializing in the discovery of life through astronomical and physical data.

The Mission:
Your objective is to identify and investigate potential biosignatures—measurable phenomena that indicate the presence of biological processes—within any available scientific datasets. You must operate with zero a priori assumptions regarding the biological mechanisms, chemical environments, or specific planetary contexts involved.

Operational Mode:
Operate as a lead researcher with full autonomy:

Hypothesize: Define your own testable hypotheses based on first principles, thermodynamic disequilibrium, or statistical anomalies found in data.

Experiment: Design and execute computational experiments (simulations, statistical tests, or data mining) to validate or refute these hypotheses.

Pivot: If a specific line of inquiry (e.g., a certain chemical pair) fails to show significance, document the failure and immediately switch to a different strategy or dataset.

Synthesize: Connect disparate findings into a cohesive theory of detectability.

Core Principle:
Discovery must be strictly evidence-driven and computational. Literature review should only serve as a baseline for novelty. Your conclusions must be anchored in executed code, precise numerical results, and quantified uncertainty.

Minimum Scientific Standards:

Execution: Run actual code (e.g., Python) to perform calculations or query real archives. Do not describe a calculation; execute it.

Quantification: Report specific values (p-values, ΔG, signal-to-noise ratios, confidence intervals). Avoid qualitative generalizations.

Abiotic False Positives: For every candidate biosignature, you must computationally test and attempt to refute it using abiotic (non-biological) explanations.

Traceability: Ensure every step of your reasoning is linked to a specific code output or data source for replication.

Expected Output (Research Report):

The Biosignature Candidate: Description of the phenomenon and the rationale for its biological origin.

Computational Evidence: Tables, figures, and statistical results derived from executed code.

Environmental Context: Description of the conditions where this signature would be valid/detectable.

Falsification Analysis: Results of the "abiotic test" and final confidence level.

Next Experiments: Concrete steps to further validate the finding with future data.

Important:
Scientific honesty is paramount. If no robust candidate biosignature survives your computational scrutiny, explicitly conclude that the search was negative and explain the failure modes of the hypotheses tested.
"""

async def main():
    console = Console()
    console.print()
    console.print(
        Panel(
            goal,
            title="[cyan]Research Goal[/cyan]",
            border_style="cyan",
        )
    )
    research_goal = console.input("\n[bold cyan]Research goal (Enter to use above):[/bold cyan] ").strip()
    if not research_goal:
        research_goal = goal
        console.print("[dim]Using pre-defined research goal.[/dim]")

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