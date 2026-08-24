"""The interview knowledge base.

Short reference notes describing what a *complete* answer to a topic covers.
This is the one genuine corpus in the system, and it is what evaluation
grounding retrieves: the evaluator is shown the reference points for the topic
so "missing concepts" means missing against something written down, not against
whatever the model happened to recall.

Notes are deliberately terse and are never shown to the candidate.
"""

from __future__ import annotations

import logging

from app.rag.indexer import Indexer
from app.rag.schemas import GLOBAL_OWNER, DocumentChunk

logger = logging.getLogger(__name__)

KNOWLEDGE_NOTES: tuple[tuple[str, str], ...] = (
    (
        "bias-variance tradeoff",
        "A complete answer separates error from bias (wrong assumptions, underfitting) "
        "from error from variance (sensitivity to the training sample, overfitting), "
        "explains why lowering one typically raises the other, and connects the "
        "tradeoff to concrete levers: model capacity, amount of data, regularisation "
        "strength, and ensembling.",
    ),
    (
        "regularization",
        "Strong answers distinguish L1 (sparsity, feature selection) from L2 (shrinkage, "
        "correlated features), note that weight decay and L2 coincide only for plain SGD "
        "and diverge for adaptive optimisers like Adam, and cover non-penalty "
        "regularisers: dropout, early stopping, data augmentation, label smoothing.",
    ),
    (
        "model evaluation metrics",
        "Expect the choice of metric to follow the cost of each error type: precision "
        "and recall over accuracy under class imbalance, PR-AUC over ROC-AUC when "
        "positives are rare, calibration when the score itself is consumed downstream, "
        "and an explicit threshold decision rather than a default 0.5.",
    ),
    (
        "overfitting and validation strategy",
        "A complete answer covers leakage first: splitting by time for temporal data, "
        "by group or user where records repeat, and fitting preprocessing inside the "
        "fold. It should mention that a validation set reused for many decisions "
        "becomes a training set, and separate model selection from final estimation.",
    ),
    (
        "feature engineering",
        "Look for treatment of high-cardinality categoricals (hashing, target encoding "
        "with out-of-fold statistics), missing-value handling as signal rather than "
        "noise, and above all train/serve consistency: the same transformation code "
        "and the same point-in-time data at training and inference.",
    ),
    (
        "optimization and learning rate schedules",
        "Strong answers cover warmup and why it stabilises early training with adaptive "
        "optimisers, cosine or linear decay, the interaction between batch size and "
        "learning rate, gradient clipping for exploding gradients, and how to read a "
        "loss curve to tell a bad learning rate from a bad initialisation.",
    ),
    (
        "normalization layers",
        "Expect batch norm's dependence on batch statistics and its train/eval "
        "discrepancy, why layer norm is used in transformers instead, the difference "
        "between pre-norm and post-norm residual blocks for training stability, and "
        "RMSNorm as a cheaper variant.",
    ),
    (
        "training instability and debugging",
        "A complete answer is procedural: overfit a single batch first, check data and "
        "label alignment, inspect gradient and activation norms, reduce the learning "
        "rate or add warmup, and look for numerical issues in mixed-precision training "
        "before blaming the architecture.",
    ),
    (
        "self-attention mechanics",
        "Expect queries, keys and values, scaling by the square root of the head "
        "dimension and why it is needed, the quadratic cost in sequence length, what "
        "multiple heads buy, and causal masking for autoregressive decoding.",
    ),
    (
        "positional encodings",
        "Strong answers contrast absolute sinusoidal and learned encodings with "
        "relative schemes, explain why RoPE is now standard and how it applies rotation "
        "in query/key space, and connect the choice to length extrapolation beyond the "
        "training context.",
    ),
    (
        "KV caching and inference cost",
        "A complete answer explains that the cache trades memory for avoided "
        "recomputation, that prefill is compute-bound while decode is memory-bandwidth-"
        "bound, how cache size scales with batch, layers, heads and context, and "
        "mitigations such as multi-query or grouped-query attention and paged attention.",
    ),
    (
        "chunking and indexing strategy",
        "Expect chunk size traded off against retrieval precision and answer "
        "completeness, overlap to avoid severing context, structure-aware splitting over "
        "fixed windows, and storing metadata for filtering. Strong answers admit the "
        "right size is an empirical question answered by evaluation.",
    ),
    (
        "embedding and retrieval quality",
        "Look for the distinction between ANN index recall (HNSW's graph parameters, "
        "IVF's probe count) and end-to-end retrieval quality, awareness that lexical "
        "search still beats embeddings for rare exact terms, and hybrid retrieval with "
        "fusion as the usual answer.",
    ),
    (
        "reranking",
        "Strong answers separate the cheap bi-encoder first stage from an expensive "
        "cross-encoder rerank over a shallow candidate set, and can reason about the "
        "latency budget that makes this two-stage design worthwhile.",
    ),
    (
        "evaluation of RAG systems",
        "A complete answer evaluates retrieval and generation separately -- recall@k and "
        "context precision for the former, faithfulness and answer relevance for the "
        "latter -- and insists on a labelled evaluation set, because 'it looks better' "
        "is not a result.",
    ),
    (
        "agent tool use and failure modes",
        "Expect concrete failure modes: looping, hallucinated tool arguments, and error "
        "cascades; plus the mitigations -- strict tool schemas, step and cost budgets, "
        "validating tool output rather than trusting it, and keeping deterministic "
        "control in code rather than in the model.",
    ),
    (
        "serving architecture and latency",
        "Strong answers separate p50 from p99, place batching and caching deliberately, "
        "distinguish online from precomputed features, and justify a hardware and "
        "autoscaling choice against a stated latency budget.",
    ),
    (
        "monitoring, drift and reliability",
        "Expect monitoring of inputs (feature drift), outputs (prediction drift) and "
        "outcomes (delayed labels), with alerting tied to business metrics, plus a "
        "rollback and retraining story and shadow or canary deployment for changes.",
    ),
)


async def seed_knowledge_base(indexer: Indexer) -> int:
    """Index the reference notes. Chunk ids are content-addressed, so this is
    safe to run on every boot -- unchanged notes are simply overwritten."""
    from app.rag.indexer import stable_chunk_id

    chunks = [
        DocumentChunk(
            chunk_id=stable_chunk_id(f"{topic}:{note}"),
            collection="knowledge",
            owner_id=GLOBAL_OWNER,
            text=f"{topic} - {note}",
            topic=topic,
            source="curated knowledge base",
        )
        for topic, note in KNOWLEDGE_NOTES
    ]
    written = await indexer.index_chunks(chunks)
    logger.info("knowledge_base_seeded notes=%d", written)
    return written
