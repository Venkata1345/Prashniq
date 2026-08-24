"""Document chunking.

Chunking exists here for one concrete reason: extraction quality and prompt
size. A long resume is split on its own structural boundaries and each chunk is
extracted independently, then merged in code. No embeddings are involved.
"""

from __future__ import annotations

import re

DEFAULT_MAX_CHARS = 1_500

# Blank lines first, then single newlines: resumes are line-oriented and often
# have no blank lines at all.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def chunk_text(text: str, *, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split on structural boundaries, packing blocks up to `max_chars`.

    Deterministic and lossless: every non-blank line of the input appears in
    exactly one chunk, in order.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    blocks = [block.strip() for block in _PARAGRAPH_SPLIT.split(text or "")]
    blocks = [block for block in blocks if block]
    if not blocks:
        return []

    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for block in blocks:
        for piece in _split_oversized(block, max_chars):
            piece_size = len(piece)
            if current and size + piece_size + 1 > max_chars:
                chunks.append("\n".join(current))
                current, size = [], 0
            current.append(piece)
            size += piece_size + 1

    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_oversized(block: str, max_chars: int) -> list[str]:
    """A single block longer than the budget is split on line boundaries, and
    only as a last resort mid-line."""
    if len(block) <= max_chars:
        return [block]

    pieces: list[str] = []
    current: list[str] = []
    size = 0

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        while len(line) > max_chars:
            if current:
                pieces.append("\n".join(current))
                current, size = [], 0
            pieces.append(line[:max_chars])
            line = line[max_chars:]
        if current and size + len(line) + 1 > max_chars:
            pieces.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1

    if current:
        pieces.append("\n".join(current))
    return pieces
