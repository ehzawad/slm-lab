# Assistant-Only Loss Masking + On-Policy Distillation on NimbusWorks (Complete)

Two techniques identified as super-important for custom agentic models, implemented end-to-end
on the NimbusWorks pipeline (Qwen3-4B QLoRA, single RTX A5000), reconciled with a Codex council
before execution and run via a five-phase workflow (prep, verify, v3 training chain, OPD, synthesis).

## What was implemented

1. **Assistant-only loss masking.** The v2 pipeline trained on the full rendered chat text -
   user turns, system prompts, and tool-result tokens all contributed loss, teaching the model
   to predict text it never generates (including hallucinating tool outputs). Fix: patched the
   Qwen3 chat template with `{% generation %}` markers around the entire assistant render
   (content + tool_calls serialization + end token - wrapping only `message.content` would mask
   tool-call syntax and defeat the tools stage), switched datasets to conversational `messages`
   format, and set `assistant_only_loss=True` (`use_liger_kernel=False`; truncation guard
   asserting no all-masked examples). Verified by a batch-level mask audit before training:
   render-equivalence with the original template, assistant+tool-call tokens trained,
   user/system/tool-result tokens masked; active-label fraction 0.870.
2. **On-policy distillation (GKD).** Final behavior-restoration stage per Thinking Machines'
   recipe: teacher = base Qwen3-4B (4-bit, frozen), student = the accumulated v3 adapter,
   `trl.experimental.gkd.GKDTrainer` with lmbda 0.5, beta 0.5, `seq_kd=False` (critical: the
   teacher knows nothing about NimbusWorks; sequence-KD would SFT on ignorant-teacher text).
   Prompt pool was behavior-only (instruction-following, refusal-framed fake entities, general
   chat; domain fact questions strictly excluded) and teacher-prefiltered: only prompts where
   the teacher actually produced the desired behavior were kept (199/200 survived).
3. **Enlarged eval sets first** (council requirement - the old sets could not detect the
   expected effect): domain 38 to 100, IF 30 to 120, halluc 20 to 120, tools 15 to 60 items.
   All baselines re-measured on the enlarged sets; only like-for-like rows are compared below.

## Results (all rows on enlarged evals)

| Row | domain_qa | instr_following | halluc_handling | tool_validity |
|-----|----------:|----------------:|----------------:|--------------:|
| base_big (untrained)   | 7.0  | 97.5 | 5.8  | 100.0 |
| v2_final_big (unmasked pipeline) | 46.0 | 97.5 | 31.7 | 100.0 |
| v3_sft       | 44.0 | 95.0 | 30.8 | 100.0 |
| v3_reasoning | 43.0 | 95.8 | 30.8 | 98.3  |
| v3_tools     | 36.0 | 99.2 | 33.3 | 100.0 |
| v3_mcp       | 51.0 | 96.7 | 35.8 | 100.0 |
| v3_dpo       | 53.0 | 96.7 | 32.5 | 100.0 |
| **v3_grpo (masked final)** | **53.0** | 95.8 | 32.5 | 100.0 |
| v3_opd (masked + GKD)      | 52.0 | 95.8 | 34.2 | 100.0 |

Note the enlarged evals reset all absolute numbers: the earlier headline 81.6 domain score was
flattered by the small 38-item set; the honest v2 number on 100 harder items is 46.0.

## Gate verdicts

- **Masking: modest win.** v3_grpo vs v2_final_big: domain +7.0 (7 items at n=100, above the
  noise band), IF -1.7 and halluc +0.8 (both within noise), tools held at 100. The literature
  predicts the gain in instruction-following, but IF was already at ceiling (97.5); the
  reclaimed capacity showed up as domain-knowledge retention instead - coherent with the
  mechanism (no gradient spent predicting user/tool-result tokens). No metric regressed
  materially, and the training objective is now correct for an agentic model (no longer
  learning to generate tool outputs).
- **OPD: wash, with the safety property proven.** v3_opd vs v3_grpo: domain -1.0, IF unchanged,
  halluc +1.7, tools held - all within noise. The behavior-only, teacher-prefiltered prompt
  design succeeded at its main risk objective (the ignorant teacher did NOT erase domain
  knowledge), but there was no behavior deficit left to repair: unlike the Thinking Machines
  scenario (IF collapsed to 45, OPD restored to 83), our masked pipeline never broke behavior
  (IF ~96 throughout). OPD is the right tool for behavior damage; this pipeline did not have any.
  No same-budget extra-training control was run, so even a positive result would have been a
  composed-recipe claim, not isolated evidence.

## Other observations

- The tools-stage dip persists in milder form (44 to 36 domain before MCP recovers to 51) even
  with replay mixing plus masking - format-narrow stages remain the pipeline's stress point.
- MCP/DPO again compounded knowledge (36 to 53), consistent with the v2 finding.
- Active-label fraction 0.870 means 13 percent of trained tokens (user/system/tool-result) no
  longer receive loss - the defect being fixed was real but modest in token share for this mix.

## Honest residuals

- v3 vs v2 also differs by retraining randomness (same steps, LRs, seeds where controllable).
- Single seed; deltas below ~3 points should be treated as noise even on the enlarged sets.
- OPD hyperparameters (lmbda 0.5, beta 0.5, 60 steps at 5e-6) were conservative; an aggressive
  sweep might show more, but the domain-erasure risk grows with strength.

## Files

`template_masked.jinja` + `mask_audit.py` (patch + verification), `gen_evals_big.py` (enlarged
evals), `train_nimbus_v3.py` (masked chain), `gen_opd_prompts.py` + `prefilter_teacher.py` +
`train_gkd.py` (OPD stage), `v3_finish.sh` (runner), `scores.json` (all rows).
