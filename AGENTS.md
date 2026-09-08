<!-- s5d:begin -->
## S5D — Agentic Change Control Plane

This repo uses **S5D** (https://github.com/system5-dev/s5d) to describe target state, record agent/tool evidence, bind architectural decisions, and verify that code still matches them.

**⛔ S5D is MANDATORY for non-trivial work.** Architectural decisions, new features, refactors >30 LOC, and any change touching multiple modules MUST go through the S5D flow before implementation. Skip ONLY for: bug fixes <30 LOC, config-only, docs-only. `S5D_BYPASS=1` is an explicit break-glass escape hatch, not routine flow; document the justification when you use it. When in doubt, run `s5d_route` to classify the request.

**Flow:** target state → edit spec → `s5d_validate` → `s5d_preview` → `s5d_approve` → run/implement → `s5d_run_gates` → `s5d_import` → `s5d_drift_check`.

**Outcome protocol.** When an approved spec declares `workflow.outcome`, treat a native host goal as a session projection only. The source of truth is `s5d outcome check <spec>` over current, source-bound gate evidence; the Stop hook enforces that verdict only on hosts that support blocking. Never auto-run authority transitions such as approval, phase acceptance, mandate admission, or waiver — they require current human authorization.

**MCP tools** (prefer over shell CLI when available):
- `s5d_route` — classify a request into tier/mode/entry
- `s5d_codebase_sync` / `s5d_codebase_check` — codebase coverage snapshot
- `s5d_discover_sync` / `s5d_discover_check` — discovery graph snapshot
- `s5d_check` — architecture check (components vs. source paths)
- `s5d_new` / `s5d_note` — create spec / quick note
- `s5d_validate` / `s5d_preview` — dry-run checks before approval
- `s5d_outcome_check` — read the shared source-bound outcome verdict
- `s5d_approve` / `s5d_import` — commit decision, bind SHA256 chain
- `s5d_drift_check` / `s5d_reconcile` / `s5d_rollback` — verify & recover
- `s5d_show` / `s5d_status` — inspect specs and project state

**Commits reference specs.** When a change is governed by an S5D spec, include `S5D-Spec: <spec-id>` as a trailer in the commit body (e.g. `S5D-Spec: feat.s5d.structure-outline-and-vertical-phases`). This binds the commit to the decision record and lets `git log --grep='S5D-Spec:'` reconstruct the architectural rationale. Trivial changes that skipped S5D need no reference.

Specs live in `.s5d/packages/`. Run `s5d --help` or read `skills/s5d/SKILL.md` for full reference.
<!-- s5d:end -->
