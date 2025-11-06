"""Text parsing and chunking functions for knowledge base processing."""

import logging
import pathlib
import re
import unicodedata
import uuid

from .config import CHUNK_CHARS, CHUNK_OVERLAP
from .utils import norm_ws, strip_noise

logger = logging.getLogger(__name__)

# Rank 23: NLTK for sentence-aware chunking
try:
    import nltk
    # Lazy download of punkt tokenizer data (only if needed)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # For newer NLTK versions
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


# ====== KB PARSING ======
def parse_articles(md_text: str) -> list:
    """Parse articles from markdown. Heuristic: '# [ARTICLE]' + optional URL line."""
    lines = md_text.splitlines()
    articles = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("# [ARTICLE]"):
            title_line = lines[i].replace("# ", "").strip()
            url = ""
            if i + 1 < len(lines) and lines[i + 1].startswith("http"):
                url = lines[i + 1].strip()
                i += 2
            else:
                i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("# [ARTICLE]"):
                buf.append(lines[i])
                i += 1
            body = "\n".join(buf).strip()
            articles.append({"title": title_line, "url": url, "body": body})
        else:
            i += 1
    if not articles:
        articles = [{"title": "KB", "url": "", "body": md_text}]
    return articles


def split_by_headings(body: str) -> list:
    """Split by H2 headers."""
    parts = re.split(r"\n(?=## +)", body)
    return [p.strip() for p in parts if p.strip()]


def sliding_chunks(text: str, maxc: int = None, overlap: int = None) -> list:
    """Overlapping chunks with sentence-aware splitting (Rank 23).

    Uses NLTK sentence tokenization to avoid breaking sentences mid-way.
    Falls back to character-based chunking if NLTK is unavailable.
    """
    if maxc is None:
        maxc = CHUNK_CHARS
    if overlap is None:
        overlap = CHUNK_OVERLAP

    out = []
    text = strip_noise(text)
    # Normalize to NFKC
    text = unicodedata.normalize("NFKC", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    if len(text) <= maxc:
        return [text]

    # Rank 23: Use sentence-aware chunking if NLTK is available
    if _NLTK_AVAILABLE:
        try:
            sentences = nltk.sent_tokenize(text)

            # Build chunks by accumulating sentences
            current_chunk = []
            current_len = 0

            for sent in sentences:
                sent_len = len(sent)

                # If single sentence exceeds maxc, fall back to character splitting
                if sent_len > maxc:
                    # Flush current chunk first
                    if current_chunk:
                        out.append(" ".join(current_chunk).strip())
                        current_chunk = []
                        current_len = 0

                    # Split long sentence by characters with consistent overlap
                    i = 0
                    while i < sent_len:
                        j = min(i + maxc, sent_len)
                        out.append(sent[i:j].strip())
                        if j >= sent_len:
                            break
                        i = j - overlap if overlap < j else 0  # FIXED: respect overlap
                    continue

                # Check if adding this sentence exceeds maxc
                if current_len + sent_len + (1 if current_chunk else 0) > maxc:
                    # Flush current chunk
                    if current_chunk:
                        out.append(" ".join(current_chunk).strip())

                    # Start new chunk with overlap (last N sentences)
                    overlap_chars = 0
                    overlap_sents = []
                    for prev_sent in reversed(current_chunk):
                        if overlap_chars + len(prev_sent) <= overlap:
                            overlap_sents.insert(0, prev_sent)
                            overlap_chars += len(prev_sent) + 1
                        else:
                            break

                    current_chunk = overlap_sents + [sent]
                    current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sent)
                    current_len += sent_len + (1 if len(current_chunk) > 1 else 0)

            # Flush final chunk
            if current_chunk:
                out.append(" ".join(current_chunk).strip())

            return out

        except Exception as e:
            # Fall back to character-based chunking if NLTK fails
            logger.warning(f"NLTK sentence tokenization failed: {e}, falling back to character chunking")

    # Fallback: Character-based chunking (original implementation)
    i = 0
    n = len(text)
    while i < n:
        j = min(i + maxc, n)
        out.append(text[i:j].strip())
        if j >= n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return out


def build_chunks(md_path: str) -> list:
    """Parse and chunk markdown."""
    raw = pathlib.Path(md_path).read_text(encoding="utf-8", errors="ignore")
    chunks = []
    for art in parse_articles(raw):
        sects = split_by_headings(art["body"]) or [art["body"]]
        for sect in sects:
            head = sect.splitlines()[0] if sect else art["title"]
            for piece in sliding_chunks(sect):
                cid = str(uuid.uuid4())
                chunks.append({
                    "id": cid,
                    "title": norm_ws(art["title"]),
                    "url": art["url"],
                    "section": norm_ws(head),
                    "text": piece
                })
    return chunks
