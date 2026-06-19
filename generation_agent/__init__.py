"""Natural-language time-series generation agent with optional MCP tools."""

from .env import load_project_env

load_project_env()

from .agent import GenerationAgent
from .dataset_generator import DatasetGenerator
from .planner import SeriesPlan
from .reference_profiler import ReferenceProfile, profile_reference_arrow, profile_reference_csv
from .semantic_types import AnomalyOverrides
from .synthesizer import synthesize_series

__all__ = [
    "AnomalyOverrides",
    "DatasetGenerator",
    "GenerationAgent",
    "ReferenceProfile",
    "SeriesPlan",
    "profile_reference_arrow",
    "profile_reference_csv",
    "synthesize_series",
]
