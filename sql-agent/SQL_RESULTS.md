# NL→SQL Agent — Quality Results (RTX A5000)

A quality-focused, **execution-verified** natural-language-to-SQL agent: Qwen3-4B + QLoRA on
`gretelai/synthetic_text_to_sql`, scored by **execution accuracy** (run the predicted SQL, compare
result sets to gold). Everything on a single A5000. Web-searched SOTA + reconciled with a Codex
council before the RL stage.

## Scorecard (150 held-out, hardened comparator)

| Model | Execution accuracy | Valid SQL | Notes |
|-------|-------------------:|----------:|-------|
| Base Qwen3-4B         | 47.3% | 95.3% | untrained |
| **SFT (direct SQL)**  | **58.0%** | **98.7%** | **+10.7 — the real win** |
| + GRPO (exec reward)  | 58.0% | 98.7% | held (safe, no gain in 80-step pilot) |
| CoT-SFT (reason+SQL)  | 45.3% | 76.0% | hurt at 4B (see below) |

## Method

- **Data:** `gretelai/synthetic_text_to_sql` (Apache-2.0), filtered to CLEAN EXECUTABLE rows
  (gold runs + returns non-empty): 3000 train / 400 eval. Each row is self-contained
  (schema + INSERT data), so execution is fully local — no external DB setup.
- **Reward + metric = execution:** build an in-memory SQLite per example, run gold vs predicted,
  compare **result sets**. `sql_exec.py`.
- **SFT:** `(schema + question → gold SQL)`, LoRA-all-linear r=16 @ 2e-4, 1 epoch.
- **GRPO:** execution reward, resumes SFT adapter, KL-anchored (β=0.04), 8 generations.

## Reconciliation with a Codex council (3/3), pre-GRPO

The council reviewed the reward/metric and RL plan and materially improved it:

- **Measured reward hack:** a probe found degenerate queries (`SELECT 1`, `SELECT *`) matched
  **12–15%** of examples — my `str(x)` comparator collapsed `1`/`1.0`/`"1"`. **Fixed:** type-aware
  comparison (numeric tolerance, NULL sentinel, string-strip) + an **anti-degenerate guard**
  (a constant/no-FROM query can't score when the gold reads tables). Degenerate match → **0%**.
- **Order-sensitivity:** compare rows in order when the gold has `ORDER BY`.
- **GRPO stability:** set explicit **KL β** (anchor to SFT so RL can't erase the SFT gain) and
  raised generations 4→8 (cuts zero-signal groups ~18%→~1.4%). Observed KL stayed ~0.001.
- **Honesty:** re-measured after the fix — the SFT 58.0% is REAL (greedy predictions never used
  the hack); the hack only mattered as an RL exploit surface, now closed.

## What worked, what didn't (honest)

- **SFT is the win** (+10.7 pts). Direct schema-conditioned SQL generation.
- **GRPO (conservative pilot) held at 58%** — KL-safe but `frac_reward_zero_std ≈ 0.5–0.6` meant
  thin signal on an already-good 4B; no gain in 80 steps. Not a failure — a known small-pilot outcome.
- **CoT-SFT hurt** (45.3%, valid-SQL 76%): the reasoning both misled the 4B (e.g. talked itself out
  of a needed `GROUP BY`) and added truncation/extraction fragility. Small-model lesson: SOTA CoT
  recipes don't automatically transfer to 4B.

## Known limitation (credibility)

Training and eval are on the **same synthetic distribution** (Gretel). The 47.3%→58.0% gain is
credible for "Gretel-shaped SQLite tasks" but not yet proof of production generalization. The right
next step is a **cross-distribution eval on Spider/BIRD dev** (needs their SQLite DBs) — flagged by
the council, deferred here due to DB-setup cost.

## To push accuracy further (for the bigger-GPU run)

1. Cross-distribution Spider/BIRD eval to make numbers production-credible.
2. Stronger GRPO: variance-selected prompts (drop all-0/all-1 groups), more steps, all 3000 prompts.
3. A larger base (Qwen3.5-9B / gpt-oss-20b) where CoT reasoning is more likely to help than hurt.
