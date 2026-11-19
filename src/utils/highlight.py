import re
from typing import List
from rich.console import Console
from rich.text import Text


def highlight_document(doc: str, keywords: List[str]):
    """
    Highlight keywords in a document using rich Text spans.
    Punctuation-aware — matches 'lucy' inside 'lucy,' 'lucy.' '(lucy)' etc.
    No ngram dependency — works for any phrase length.
    """
    # Sort by length desc — match longest phrase first ("skin condition" before "skin")
    sorted_keywords = sorted(keywords, key=len, reverse=True)

    # Build one regex pattern: \b word boundary aware, ignore case
    # re.escape handles special chars in keyword (e.g. "Medi-Cal", "C++")
    pattern = "|".join(
        r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
        for kw in sorted_keywords
        if kw.strip()
    )

    if not pattern:
        Console().print(doc)
        return

    # Use rich Text for span-based styling — no markup injection
    text = Text(doc)
    for match in re.finditer(pattern, doc, flags=re.IGNORECASE):
        text.stylize("black on #FFFF00", match.start(), match.end())

    Console().print(text)