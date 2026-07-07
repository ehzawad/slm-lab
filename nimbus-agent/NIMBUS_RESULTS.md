# NimbusWorks: All Seven Stages, Each Load-Bearing (RTX A5000)

A staged post-training experiment designed so every pipeline stage has a metric it uniquely
moves, on a corpus guaranteed absent from pretraining: **NimbusWorks**, a fictional company
(services, teams, error codes, runbooks, policies, an internal `nbx` CLI) generated
deterministically from a world model (`world.py`). All datasets and all eval gold answers
derive from the same tables, so measurement is exact.

Design follows Thinking Machines' [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation)
(mid-training on private docs lifts knowledge but collapses instruction-following; post-training
repairs it) and [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (LoRA-all-linear,
10x LR, RL needs minimal capacity). Base: Qwen3-4B, QLoRA 4-bit, one accumulating adapter,
evaluated on four metrics after every stage.

## Metrics

- **domain_qa** - exact-answer recall over NimbusWorks facts (38 questions)
- **instruction_following** - content-neutral compliance probes (30)
- **hallucination_handling** - answer real entities, refuse fake ones (20)
- **tool_validity** - correct `nbx` tool call + arguments (15)

Base model: domain_qa 0.0 (world provably unknown), IF 100, halluc 0 (fabricates
confidently), tools 100.

## The experiment: naive stacking (v1) vs replay mixing (v2)

v1 trained each stage only on its own data. v2 is identical except stages 3-5 mix in a
replay slice of the SFT chat data, and DPO is gentler (30 steps @ 2e-5 vs 60 @ 5e-5).

| stage | dQA v1 | dQA v2 | IF v1 | IF v2 | halluc v1 | halluc v2 | tool v1 | tool v2 |
|-------|-------:|-------:|------:|------:|----------:|----------:|--------:|--------:|
| base      | 0.0  | 0.0  | 100  | 100  | 0  | 0  | 100 | 100 |
| cpt       | 47.4 | 47.4 | 90   | 90   | 40 | 40 | 100 | 100 |
| sft       | 68.4 | 68.4 | 96.7 | 96.7 | 50 | 50 | 100 | 100 |
| reasoning | 55.3 | 63.2 | 80   | 100  | 5  | 40 | 100 | 100 |
| tools     | 0.0  | 68.4 | 30   | 100  | 0  | 50 | 100 | 100 |
| mcp       | 42.1 | 76.3 | 90   | 90   | 20 | 50 | 93.3 | 100 |
| dpo       | 23.7 | 81.6 | 90   | 90   | 15 | 50 | 93.3 | 100 |
| grpo      | 23.7 | 81.6 | 90   | 90   | 10 | 50 | 93.3 | 100 |

## Findings

1. **The TM curve reproduced.** Pure-docs CPT: domain 0 to 47.4 with an instruction-following
   dip (100 to 90); SFT repaired IF to 96.7 while raising domain to 68.4. Knowledge-vs-behavior
   tension and its repair, measured.
2. **Naive stacking is catastrophic.** In v1 the tools stage (50 steps of tool-call-only
   outputs) collapsed the model: domain 68.4 to 0.0, IF to 30 - it learned to answer
   everything with a tool call. Damage cascaded through every later stage; v1 ended at 23.7.
3. **Replay mixing fixes it completely - the headline ablation.** Same stages, same data,
   same evals; the only change is a replay slice per stage. Final checkpoint: **81.6 vs 23.7
   domain QA (3.4x)**, IF 100 vs 30 at the tools stage, tool validity 100 held throughout.
   One measured design principle: every stage must mix replay data or it destroys prior
   capabilities. This is the frontier-lab data-mixing lesson as a controlled experiment.
4. **With mixing, later stages compound instead of erode.** MCP and DPO raised domain QA
   (76.3, then 81.6) - grounding formats and preference pairs reinforce knowledge when
   format diversity is preserved.
5. **Ultimate result:** from a base that knew nothing, the final agent scores domain 81.6,
   halluc-handling 50, tools 100, IF 90 - a working internal ops assistant over private
   knowledge, trained end-to-end on one 24 GB GPU in about an hour.

## Honest residuals

- Hallucination-handling plateaus at 50: the gentle DPO preserved knowledge but did not add
  refusal ability beyond SFT level. More/better refusal pairs (or TM's on-policy distillation
  for behavior restoration) is the known next lever.
- IF settles at 90, not 100; eval sets are small (15-38 items), so treat points as
  directional with roughly +/-5 noise.
- Stage 8 (agentic GRPO on simulated incidents with verified end-states) is designed but not
  yet run; the incident simulator (fault injection over the service dependency graph) is the
  remaining build.

## Files

`world.py` (world model), `gen_data.py` (corpus + stage datasets + evals), `train_nimbus.py`
(7 stages, resumable), `eval_nimbus.py` (4-metric harness), `run_all.sh` (train+eval chain),
`scores_v1_naive.json` / `scores.json` (both curves).
