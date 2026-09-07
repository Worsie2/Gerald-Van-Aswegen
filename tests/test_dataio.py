"""dataio.loaders: the SQL read-only guard and connection-string redaction.

Both are security boundaries with an explicit promise attached ("this
platform reads data, it does not modify the user's database") and neither
had any test coverage before this file.
"""

from __future__ import annotations

import sqlite3

import pytest

from dsai.dataio.loaders import LoadError, _extra_statements, _redact, load_sql


# ---------------------------------------------------------------------------
# _extra_statements — the multi-statement detector
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "SELECT * FROM customers",
    "SELECT * FROM customers;",
    "  select * from customers  ",
    "WITH recent AS (SELECT 1) SELECT * FROM recent",
    "SELECT 'a;b' AS x",                       # semicolon inside a string literal
    'SELECT "weird;column" FROM t',            # semicolon inside a quoted identifier
    "SELECT 'it''s fine; really' FROM t",      # escaped quote followed by a literal ';'
])
def test_extra_statements_false_for_single_statement(query):
    assert _extra_statements(query) is False


@pytest.mark.parametrize("query", [
    "SELECT * FROM customers; DROP TABLE customers;",
    "SELECT 1; DELETE FROM customers",
    "SELECT 1;SELECT 2",
])
def test_extra_statements_true_for_chained_statements(query):
    assert _extra_statements(query) is True


# ---------------------------------------------------------------------------
# load_sql — keyword guard and multi-statement guard, against a real database
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_db(tmp_path):
    db_path = tmp_path / "guard.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE customers (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO customers VALUES (1, 'a'), (2, 'b')")
    conn.commit()
    conn.close()
    return f"sqlite:///{db_path}"


def test_load_sql_accepts_a_plain_select(sqlite_db):
    frame, source = load_sql("SELECT * FROM customers", sqlite_db)
    assert len(frame) == 2
    assert source.kind == "sql"


@pytest.mark.parametrize("query", [
    "DROP TABLE customers",
    "DELETE FROM customers",
    "UPDATE customers SET name = 'x'",
    "INSERT INTO customers VALUES (3, 'c')",
    "ALTER TABLE customers ADD COLUMN x INT",
    "ATTACH DATABASE 'other.db' AS other",
])
def test_load_sql_rejects_writes(sqlite_db, query):
    with pytest.raises(LoadError, match="read"):
        load_sql(query, sqlite_db)


def test_load_sql_rejects_chained_statements_even_when_first_is_a_read(sqlite_db):
    # The keyword check alone would let this through — it starts with SELECT.
    # Some backends (Postgres via psycopg2, notably) execute every
    # semicolon-separated statement in one call, so the second, destructive
    # statement must never reach the database.
    with pytest.raises(LoadError, match="single statement"):
        load_sql("SELECT * FROM customers; DROP TABLE customers;", sqlite_db)

    # Confirm the table really does still exist and hold both rows.
    conn = sqlite3.connect(sqlite_db.replace("sqlite:///", ""))
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 2
    conn.close()


def test_load_sql_allows_one_harmless_trailing_semicolon(sqlite_db):
    frame, _ = load_sql("SELECT * FROM customers;", sqlite_db)
    assert len(frame) == 2


# ---------------------------------------------------------------------------
# _redact — connection strings must never carry credentials once stored
# ---------------------------------------------------------------------------

def test_redact_strips_username_and_password():
    assert _redact("postgresql://user:secret@localhost:5432/mydb") == \
        "postgresql://localhost:5432/mydb"


def test_redact_leaves_a_credential_free_string_unchanged():
    assert _redact("sqlite:///local.db") == "sqlite:///local.db"


def test_redact_leaves_a_schemeless_string_unchanged():
    assert _redact("not-a-url") == "not-a-url"
