"""Paper2Bench: Convert academic papers into benchmark tasks for AI agents."""

from .api import BenchmarkResult, run
from .benchmark_generator import generate_variants
from .instruction_generator import extract_yaml_from_pdf, render_instruction
from .llm import chat
from .paper_classifier import classify_paper
from .paper_spec import extract_paper_spec
from .plan_generator import build_instruction_gt, generate_supplementary_plan
from .tree_parser import parse_paper_to_tree
from .verify import verify_instruction_gt

__version__ = "0.3.0"

__all__ = [
    "run",
    "BenchmarkResult",
    "parse_paper_to_tree",
    "classify_paper",
    "extract_yaml_from_pdf",
    "render_instruction",
    "generate_supplementary_plan",
    "build_instruction_gt",
    "verify_instruction_gt",
    "extract_paper_spec",
    "generate_variants",
    "chat",
]
