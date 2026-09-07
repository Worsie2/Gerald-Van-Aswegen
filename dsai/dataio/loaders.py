"""Dataset ingestion.

One entry point, :func:`load_dataset`, that accepts a path, file-like object or
SQL connection and returns a DataFrame plus a provenance record describing
exactly where the data came from (needed for reproducibility).
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO

import pandas as pd

from dsai.core.schema import JsonMixin

SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "csv",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xls": "excel",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".feather": "feather",
    ".sav": "spss",
    ".dta": "stata",
    ".sas7bdat": "sas",
    ".pkl": "pickle",
    ".gz": "csv",
    ".zip": "csv",
}


@dataclass
class DataSource(JsonMixin):
    """Where a dataset came from — recorded so a run can be replayed."""

    kind: str                      # file | sql | dataframe
    name: str = ""
    path: str | None = None
    file_format: str | None = None
    sheet: str | None = None
    query: str | None = None
    connection: str | None = None  # redacted connection string
    n_rows: int = 0
    n_columns: int = 0
    content_hash: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


class LoadError(RuntimeError):
    """Raised when a dataset cannot be read, with an actionable message."""


def _redact(conn: str) -> str:
    """Strip credentials out of a connection string before storing it."""
    if "://" not in conn:
        return conn
    scheme, rest = conn.split("://", 1)
    if "@" in rest:
        rest = rest.split("@", 1)[1]
    return f"{scheme}://{rest}"


def _hash_frame(frame: pd.DataFrame) -> str:
    try:
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
        )
        digest.update(",".join(map(str, frame.columns)).encode())
        return digest.hexdigest()[:16]
    except Exception:
        return hashlib.sha256(repr(frame.shape).encode()).hexdigest()[:16]


def detect_format(path: str | Path) -> str:
    suffixes = Path(path).suffixes
    for suffix in reversed(suffixes):
        fmt = SUPPORTED_EXTENSIONS.get(suffix.lower())
        if fmt:
            return fmt
    raise LoadError(
        f"Cannot tell what format '{Path(path).name}' is. "
        f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    )


def _sniff_separator(sample: str) -> str:
    candidates = {",": sample.count(","), ";": sample.count(";"), "\t": sample.count("\t"), "|": sample.count("|")}
    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 0 else ","


def read_csv(source: str | Path | IO[bytes], **kwargs: Any) -> pd.DataFrame:
    """Read a delimited text file, sniffing the separator and encoding."""
    read_kwargs = {"low_memory": False, **kwargs}
    if "sep" not in read_kwargs:
        try:
            if hasattr(source, "read"):
                pos = source.tell()
                head = source.read(65536)
                source.seek(pos)
                text = head.decode("utf-8", errors="replace") if isinstance(head, bytes) else head
            else:
                with open(source, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read(65536)
            read_kwargs["sep"] = _sniff_separator(text)
        except Exception:
            read_kwargs["sep"] = ","
    for encoding in (read_kwargs.pop("encoding", None), "utf-8", "utf-8-sig", "latin-1"):
        if encoding is None:
            continue
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding, **read_kwargs)
        except UnicodeDecodeError:
            continue
    raise LoadError("Could not decode the file with UTF-8 or Latin-1. Re-export it as UTF-8 CSV.")


def read_json_any(source: Any, **kwargs: Any) -> pd.DataFrame:
    """Read JSON, coping with records, a dict of columns, or a nested payload."""
    try:
        return pd.read_json(source, **kwargs)
    except ValueError:
        import json

        if hasattr(source, "seek"):
            source.seek(0)
            payload = json.load(source)
        else:
            with open(source, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            # find the first list-of-dicts value and flatten it
            for key, value in payload.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    return pd.json_normalize(value)
            return pd.json_normalize(payload)
        raise LoadError("The JSON file does not contain a table-shaped structure.")


def excel_sheet_names(path: str | Path | IO[bytes]) -> list[str]:
    return list(pd.ExcelFile(path).sheet_names)


def load_file(
    path: str | Path | IO[bytes],
    file_format: str | None = None,
    name: str | None = None,
    sheet: str | int | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, DataSource]:
    """Load a dataset from a path or an uploaded file-like object."""
    display_name = name or getattr(path, "name", None) or str(path)
    fmt = file_format or detect_format(display_name)

    readers = {
        "csv": lambda: read_csv(path, **kwargs),
        "tsv": lambda: read_csv(path, sep="\t", **kwargs),
        "excel": lambda: pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, **kwargs),
        "json": lambda: read_json_any(path, **kwargs),
        "jsonl": lambda: pd.read_json(path, lines=True, **kwargs),
        "parquet": lambda: pd.read_parquet(path, **kwargs),
        "feather": lambda: pd.read_feather(path, **kwargs),
        "stata": lambda: pd.read_stata(path, **kwargs),
        "spss": lambda: pd.read_spss(path, **kwargs),
        "sas": lambda: pd.read_sas(path, **kwargs),
        "pickle": lambda: pd.read_pickle(path, **kwargs),
    }
    if fmt not in readers:
        raise LoadError(f"'{fmt}' files are not supported yet.")
    try:
        frame = readers[fmt]()
    except ImportError as exc:
        raise LoadError(
            f"Reading {fmt} files needs an extra package that is not installed: {exc}."
        ) from exc
    except FileNotFoundError as exc:
        raise LoadError(f"File not found: {path}") from exc
    except Exception as exc:
        raise LoadError(f"Could not read {display_name} as {fmt}: {exc}") from exc

    if isinstance(frame, dict):  # multi-sheet Excel
        first = next(iter(frame))
        sheet = first
        frame = frame[first]

    frame = normalise_columns(frame)
    source = DataSource(
        kind="file",
        name=Path(str(display_name)).name,
        path=str(path) if not hasattr(path, "read") else None,
        file_format=fmt,
        sheet=str(sheet) if sheet is not None else None,
        n_rows=len(frame),
        n_columns=int(frame.shape[1]),
        content_hash=_hash_frame(frame),
        options={k: str(v) for k, v in kwargs.items()},
    )
    return frame, source


def _extra_statements(query: str) -> bool:
    """True if ``query`` holds more than one SQL statement.

    Checking only the opening keyword lets ``SELECT 1; DROP TABLE x;`` through
    unnoticed on any backend whose driver executes semicolon-separated
    statements in one call (Postgres via psycopg2 does, by default). This
    scans for a statement-separating ``;`` outside of quoted string literals,
    ignoring one optional trailing ``;`` at the very end of the query.
    """
    body = query.strip()
    if body.endswith(";"):
        body = body[:-1]
    in_single = in_double = False
    for char in body:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == ";" and not in_single and not in_double:
            return True
    return False


def load_sql(
    query: str,
    connection_string: str,
    name: str = "sql_query",
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, DataSource]:
    """Run a read query against any SQLAlchemy-supported database.

    Only a single SELECT/WITH statement is accepted — this platform reads
    data, it does not modify the user's database.
    """
    stripped = query.strip().lstrip("(").lstrip()
    if not stripped.lower().startswith(("select", "with", "show", "pragma", "describe")):
        raise LoadError(
            "Only read queries are allowed here (SELECT / WITH). "
            "Run any data-modifying statement in your own database client."
        )
    if _extra_statements(query):
        raise LoadError(
            "Only a single statement is allowed here. Remove anything after the "
            "first query — chaining statements with ';' is not permitted, even "
            "read-only ones, since some database drivers will run all of them."
        )
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:  # pragma: no cover
        raise LoadError("SQL loading needs SQLAlchemy: pip install sqlalchemy") from exc
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            frame = pd.read_sql(text(query), conn, params=params)
    except Exception as exc:
        raise LoadError(f"Database query failed: {exc}") from exc

    frame = normalise_columns(frame)
    return frame, DataSource(
        kind="sql",
        name=name,
        query=query,
        connection=_redact(connection_string),
        n_rows=len(frame),
        n_columns=int(frame.shape[1]),
        content_hash=_hash_frame(frame),
    )


def list_sql_tables(connection_string: str) -> list[str]:
    from sqlalchemy import create_engine, inspect

    engine = create_engine(connection_string)
    return list(inspect(engine).get_table_names())


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Make column names usable: strings, stripped, and unique."""
    frame = frame.copy()
    seen: dict[str, int] = {}
    names: list[str] = []
    for i, col in enumerate(frame.columns):
        name = str(col).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    frame.columns = names
    return frame


def load_dataset(source: Any, **kwargs: Any) -> tuple[pd.DataFrame, DataSource]:
    """Universal loader: DataFrame, path, file-like object, or (query, conn) tuple."""
    if isinstance(source, pd.DataFrame):
        frame = normalise_columns(source)
        return frame, DataSource(
            kind="dataframe",
            name=kwargs.get("name", "in_memory"),
            n_rows=len(frame),
            n_columns=int(frame.shape[1]),
            content_hash=_hash_frame(frame),
        )
    if isinstance(source, tuple) and len(source) == 2:
        return load_sql(source[0], source[1], **kwargs)
    return load_file(source, **kwargs)
