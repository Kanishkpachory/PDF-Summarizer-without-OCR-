import re
import os
from typing import List, Dict, Any


def _count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())


def _looks_like_noise(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True

    # Pure numbers or short numeric tokens (axis ticks)
    if re.fullmatch(r"[\d\s,.\-]+", s) and len(s) <= 20:
        return True

    # Common footer/header noise patterns
    if "DIW Weekly Report" in s or "© DIW" in s or "doi:" in s.lower():
        return True

    return False


def create_chunks(components: List[Dict[str, Any]], graphics: List[Dict[str, Any]], tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Better chunking:
    - Start a new chunk when we hit a 'title' or obvious heading keyword (ABSTRACT, Conclusion, Figure)
    - Keep bullets together
    - Track page numbers for each chunk
    """
    chunks: List[Dict[str, Any]] = []
    current = {"page": 1, "title": "Slide", "content": "", "summary": "", "questions": []}

    def flush():
        if current["content"].strip():
            word_count = _count_words(current["content"])
            # Only keep chunks with at least 150 words
            if word_count >= 150:
                current["questions"] = generate_questions(current["content"])
                # Mark chunks > 200 words for summarization (summary will be added by API)
                if word_count > 200:
                    current["summary"] = f"[To be summarized - {word_count} words]"
                chunks.append(dict(current))

    for c in components:
        t = c.get("type", "paragraph")
        text = (c.get("content") or "").strip()
        page = c.get("page", current["page"])  # Get page from component or use current

        if _looks_like_noise(text):
            continue

        # Start new slide on titles or common section labels
        is_heading = (t == "title") or text.upper() in {"ABSTRACT", "CONCLUSION", "PRODUCTIVITY"} or text.startswith("Figure")
        if is_heading and current["content"].strip():
            flush()
            current["page"] = page  # Update page for new chunk
            current["title"] = text[:80]
            current["content"] = ""
            current["summary"] = ""
            current["questions"] = []
            continue
        
        # Update page if it changes
        if page != current["page"]:
            current["page"] = page

        # Format content nicely
        prefix = "- " if text.startswith("•") else ""
        cleaned = text.lstrip("•").strip() if text.startswith("•") else text
        current["content"] += f"{prefix}{cleaned}\n"

        # Limit per slide (so you don’t get 90 slides)
        if len(current["content"].splitlines()) >= 10:
            flush()
            current["title"] = "Continued"
            current["content"] = ""
            current["summary"] = ""
            current["questions"] = []

    flush()
    return chunks


def generate_questions(content: str) -> List[str]:
    # Simple but less repetitive questions
    lines = [ln.strip("- ").strip() for ln in content.splitlines() if ln.strip()]
    focus = (lines[0] if lines else content[:60])

    return [
        f"Summarize the key message of: {focus}",
        "Name two important facts/arguments mentioned.",
        "Explain one implication for policy or practice.",
    ]