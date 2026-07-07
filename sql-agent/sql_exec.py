"""Execution utilities for the NL->SQL agent (gretelai/synthetic_text_to_sql).
Each example is self-contained: sql_context = CREATE TABLE + INSERT data. We build an
in-memory SQLite per example, execute gold vs predicted SQL, and compare RESULT SETS —
a clean execution-based reward and eval metric. The model is shown schema-only (DDL);
execution uses the full context (schema + data)."""
import sqlite3, re

def build_db(context):
    conn = sqlite3.connect(":memory:")
    conn.text_factory = str
    conn.executescript(context)  # CREATE TABLE ... ; INSERT INTO ... ;
    return conn

def schema_ddl(context):
    """Only the CREATE TABLE statements (what the model is allowed to see)."""
    return " ".join(s.strip() + ";" for s in context.split(";")
                    if s.strip().lower().startswith("create table"))

def run(conn, sql):
    return conn.execute(sql).fetchall()

_NULL = "\x00NULL\x00"

def _cell(x):
    """Type-aware normalization: numbers compared numerically (int 1 == float 1.0),
    NULL as a sentinel (not the string 'None'), strings stripped and type-tagged so a
    numeric 1 never equals the string '1' (which enabled the SELECT 1 reward hack)."""
    if x is None:
        return _NULL
    if isinstance(x, bool):
        return f"n:{int(x)}"
    if isinstance(x, (int, float)):
        return f"n:{round(float(x), 6)}"
    return "s:" + str(x).strip()

def _rows(rows):
    return [tuple(_cell(x) for x in r) for r in rows]

def result_match(a, b, ordered=False):
    ra, rb = _rows(a), _rows(b)
    if ordered:              # order matters when the gold query ranks (ORDER BY)
        return ra == rb
    return sorted(ra, key=repr) == sorted(rb, key=repr)

def _gold_ordered(sql):
    return " order by " in f" {sql.lower()} "

def _has_from(sql):
    return re.search(r"\bfrom\b", sql, re.I) is not None

def gold_is_valid(context, gold_sql):
    """Keep only examples whose gold executes cleanly and returns non-empty rows."""
    try:
        c = build_db(context); res = run(c, gold_sql); c.close()
        return len(res) > 0
    except Exception:
        return False

def clean_sql(text):
    """Extract a SQL statement from model output (strip fences/prose)."""
    if "```" in text:
        m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
        if m: text = m.group(1)
    m = re.search(r"\b(SELECT|WITH)\b.*", text, re.S | re.I)
    text = m.group(0) if m else text
    text = text.split(";")[0].strip()  # first statement
    return text

def exec_reward(context, gold_sql, pred_text):
    """1.0 if predicted SQL's result set matches gold's, else 0.0 (errors -> 0).
    Anti-degenerate guard: a constant query (no FROM) can't earn reward when the gold
    reads tables — kills the SELECT 1 / SELECT 'x' hack the council measured at ~12-15%.
    Order-sensitive when the gold query ranks (ORDER BY)."""
    pred = clean_sql(pred_text)
    if not pred:
        return 0.0
    if _has_from(gold_sql) and not _has_from(pred):
        return 0.0
    try:
        c = build_db(context)
        gold = run(c, gold_sql)
        pred_res = run(c, pred)
        c.close()
        return 1.0 if result_match(gold, pred_res, ordered=_gold_ordered(gold_sql)) else 0.0
    except Exception:
        return 0.0

PROMPT = ("You are a SQLite expert. Given the schema, write ONE SQLite query that answers "
          "the question. Reply with only the SQL query, no explanation.\n\n"
          "Schema:\n{schema}\n\nQuestion: {question}")

def make_prompt(context, question):
    return PROMPT.format(schema=schema_ddl(context), question=question)

COT_PROMPT = ("You are a SQLite expert. Given the schema, reason briefly about which tables and "
              "columns are needed, then write ONE SQLite query.\n"
              "Format exactly as:\nReasoning: <short reasoning>\nSQL: <query>\n\n"
              "Schema:\n{schema}\n\nQuestion: {question}")

def make_cot_prompt(context, question):
    return COT_PROMPT.format(schema=schema_ddl(context), question=question)
