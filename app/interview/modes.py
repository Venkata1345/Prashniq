"""Interview modes as configuration, not conditionals.

Adding a mode means adding a row here. Behaviour that genuinely differs per
mode (topics, scoring dimensions, interviewer framing, budgets) is data;
control flow stays shared.
"""

from __future__ import annotations

from dataclasses import dataclass

CORE_DIMENSIONS = ("technical_correctness", "technical_depth", "communication")


@dataclass(frozen=True)
class InterviewMode:
    key: str
    display_name: str
    topics: tuple[str, ...]
    dimensions: tuple[str, ...]
    focus: str
    max_questions: int = 8
    requires_candidate_context: bool = False
    max_follow_up_depth: int = 2
    starting_difficulty: int = 3
    time_limit_seconds: int = 45 * 60

    @property
    def uses_blueprint_topics(self) -> bool:
        """Modes with no standing topic list get every topic from the
        blueprint built out of the candidate's resume and target role."""
        return not self.topics

    @property
    def extra_dimensions(self) -> tuple[str, ...]:
        return tuple(d for d in self.dimensions if d not in CORE_DIMENSIONS)


_MODES: dict[str, InterviewMode] = {
    mode.key: mode
    for mode in (
        InterviewMode(
            key="ml_fundamentals",
            display_name="AI/ML Fundamentals",
            topics=(
                "bias-variance tradeoff",
                "regularization",
                "model evaluation metrics",
                "feature engineering",
                "overfitting and validation strategy",
            ),
            dimensions=CORE_DIMENSIONS + ("reasoning", "tradeoff_awareness"),
            focus="core machine learning theory and practical modelling judgement",
        ),
        InterviewMode(
            key="deep_learning",
            display_name="Deep Learning",
            topics=(
                "optimization and learning rate schedules",
                "normalization layers",
                "regularization in deep nets",
                "training instability and debugging",
                "architecture choices",
            ),
            dimensions=CORE_DIMENSIONS + ("reasoning", "tradeoff_awareness"),
            focus="deep neural network training, debugging and architecture decisions",
        ),
        InterviewMode(
            key="transformers",
            display_name="Transformers",
            topics=(
                "self-attention mechanics",
                "positional encodings",
                "KV caching and inference cost",
                "pretraining vs fine-tuning",
                "context length scaling",
            ),
            dimensions=CORE_DIMENSIONS + ("reasoning", "tradeoff_awareness"),
            focus="transformer internals and the cost model of training and inference",
        ),
        InterviewMode(
            key="rag_agents",
            display_name="RAG and Agents",
            topics=(
                "chunking and indexing strategy",
                "embedding and retrieval quality",
                "reranking",
                "evaluation of RAG systems",
                "agent tool use and failure modes",
            ),
            dimensions=CORE_DIMENSIONS + ("reasoning", "tradeoff_awareness"),
            focus="retrieval-augmented generation and agentic system design",
        ),
        InterviewMode(
            key="ml_system_design",
            display_name="ML System Design",
            topics=(
                "requirements and constraints",
                "data pipeline and features",
                "serving architecture and latency",
                "scalability and cost",
                "monitoring, drift and reliability",
            ),
            dimensions=(
                "requirements_clarification",
                "architecture",
                "scalability",
                "reliability",
                "tradeoff_reasoning",
                "ml_ai_understanding",
                "communication",
            ),
            focus="end-to-end design of production ML systems under real constraints",
            max_questions=6,
            max_follow_up_depth=3,
        ),
        InterviewMode(
            key="resume_deep_dive",
            display_name="Resume Deep Dive",
            topics=(),
            dimensions=CORE_DIMENSIONS + ("reasoning", "tradeoff_awareness"),
            focus=(
                "the systems the candidate claims to have built, probed for depth "
                "of ownership and real understanding"
            ),
            requires_candidate_context=True,
            max_follow_up_depth=3,
        ),
        InterviewMode(
            key="jd_targeted",
            display_name="Role-Targeted Interview",
            topics=(),
            dimensions=CORE_DIMENSIONS + ("reasoning", "tradeoff_awareness"),
            focus="the skills this specific role requires, in priority order",
            requires_candidate_context=True,
        ),
    )
}

DEFAULT_MODE = "ml_fundamentals"


class UnknownInterviewMode(KeyError):
    pass


def get_mode(key: str) -> InterviewMode:
    try:
        return _MODES[key]
    except KeyError as exc:
        raise UnknownInterviewMode(
            f"unknown interview type '{key}'; available: {sorted(_MODES)}"
        ) from exc


def available_modes() -> list[InterviewMode]:
    return list(_MODES.values())
