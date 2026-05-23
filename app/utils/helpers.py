import csv
import io
from datetime import datetime


def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime(fmt)


def history_to_csv(history: list[dict]) -> str:
    """
    Serialize a list of conversation-history dicts to a CSV string.
    The fieldnames are taken from the first record's keys.
    """
    if not history:
        return ""
    fieldnames = list(history[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(history)
    return buffer.getvalue()


def safe_decode(content: bytes, encoding: str = "utf-8") -> str:
    """Decode bytes to str, replacing invalid byte sequences."""
    return content.decode(encoding, errors="replace")
