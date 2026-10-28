# Bundled Skills & Commands: `/loop`, `/goal`, `/batch`
 
| | For | Mechanism | Claude Code | Codex |
| --- | --- | --- | --- | --- |
| `/loop` | Repeat work or checks over time | Schedules recurring prompts, optionally rerunning another Skill | Bundled Skill (alias `/proactive`) | — |
| `/goal` | Keep working toward a defined outcome | Preserves a completion condition; continues across turns | Built-in command | `/goal` (first here) |
| `/batch` | Split a large change into parallel work | Decomposes work, one worker per unit in its own Git worktree | Bundled Skill | — |
 
## `/loop`
 
A cron scheduler inside your local Claude Code session, not on Anthropic's servers.
 
| You give it | Example | What happens |
| --- | --- | --- |
| Interval + prompt | `/loop 5m check the deploy` | Fixed schedule |
| Prompt only | `/loop check CI and address any new review comments` | Claude picks the interval each iteration, 1 min–1 h, and can stop itself when done |
| Nothing / interval only | `/loop` · `/loop 15m` | Built-in maintenance prompt, or your `loop.md` |
 
- Units `s`, `m`, `h`, `d`; seconds round up to a minute; `7m`/`90m` round to the nearest clean cron step.
- Adaptive scheduling, not continuous reasoning: each iteration finishes, then schedules the next. It isn't watching in between (Monitor tool can stream events instead of polling where available).
- Requires a running session (`/background` keeps it alive without a terminal). Recurring tasks expire after 7 days. Self-paced loops aren't restored on `--resume`; fixed-interval ones are. `Esc` stops a self-paced loop.
- With another Skill: `/loop 20m /review-pr 1234`. Only Skills Claude may invoke itself run — `disable-model-invocation: true` (incl. bundled `/verify`), `skillOverrides`-hidden Skills, built-in commands and MCP prompts arrive as plain text.
- `.claude/loop.md` (project) or `~/.claude/loop.md` (user) replaces the built-in maintenance prompt for bare `/loop`. Plain Markdown, no structure; edits apply next iteration; >25,000 bytes truncated; ignored when you pass a prompt.
Claude Code team examples (power-user guide): `/loop 5m /babysit` (PR reviews, feedback, rebases) · `/loop 30m /slack-feedback` (Slack feedback → PRs) · `/loop 1h /pr-pruner` (close stale PRs).
 
### `/loop` vs cloud vs Desktop
 
| | `/loop` | Cloud (Routines, `/schedule`) | Desktop scheduled task |
| --- | --- | --- | --- |
| Runs on | Your machine | Anthropic cloud, fresh clone each run | Your machine |
| Needs open session / machine on | Yes / Yes | No / No | No / Yes |
| Minimum interval | 1 min | 1 h | 1 min |
| Local files | Yes | No | Yes |
| Permission prompts | Inherits session | None (autonomous) | Per task |
| Triggers | Time | Time, API call, GitHub events | Time |
 
`/loop` is a scheduling mechanism, not loop engineering: it doesn't give you independent verification, progress tracking or stopping criteria.
 
## `/goal`
 
A plan chooses the implementation path; a goal preserves the desired outcome. Plan-only work doesn't trigger continuation (Codex).
 
| | Plan | Goal |
| --- | --- | --- |
| Purpose | Agree on how to approach work | Keep working toward an outcome |
| Useful when | Investigating choices, aligning on implementation | Repeated investigation, execution and verification |
| Approach can change? | Yes, with plan revisions | Yes, as the agent learns |
| Human involvement | Planning/approval stage | Depends on constraints, blockers, outcome type |
| Example | How should we implement subscription upgrades? | Complete the approved subscription upgrade work and verify it |
 
### Codex
 
```text
/goal Reduce p95 checkout latency below 120 ms
on the checkout benchmark while keeping the
correctness suite green.
```
 
| Command | |
| --- | --- |
| `/goal` | View current Goal |
| `/goal pause` / `/goal resume` | Pause / resume |
| `/goal clear` | Remove |
 
- Persisted thread state (objective, lifecycle, budget, progress) — not global memory, not project instructions.
- Continues only at safe boundaries: thread idle, Goal active, within budget, nothing pending, no queued input. Continuation prompts re-orient Codex on the objective and require comparing progress against concrete evidence.
### Claude Code
 
```text
/goal all tests in test/auth pass and lint is clean
```
 
- Wrapper around a session-scoped prompt-based Stop hook. After each turn the small fast model (`ANTHROPIC_DEFAULT_HAIKU_MODEL`) returns **not yet met** (keep going, reason is guidance) / **met** (clear, record achieved) / **impossible** (clear, record failed).
- The evaluator doesn't run commands or read files — it judges only what Claude surfaced in the conversation. If Claude misreports a test result, the evaluator won't catch it by rerunning. Write conditions Claude's output can demonstrate: one measurable end state, a stated check (`npm test exits 0`), constraints, and a bound (`or stop after 20 turns`).
- `/goal` = status (condition, elapsed, turns, tokens, last reason); `/goal clear` (aliases `stop`, `off`, `reset`, `none`, `cancel`). No pause/resume — pauses itself on rate limits, retries transient errors (v2.1.269+). Stops with a warning if several turns pass with no tool use.
- One goal per session; condition ≤ 4,000 chars; restored on resume; `claude -p "/goal …"` runs to completion non-interactively. Doesn't change permission mode — pair with auto mode to run unattended.
### Reward hacking
 
A poorly defined goal is met at the expense of the objective: remove validation, skip expensive operations, return incomplete results because that makes the endpoint faster. If the same agent writes the acceptance criteria and the implementation, it narrows success to what's easy.
 
SpecBench (arXiv 2605.21384, May 2026): agents given specs + visible tests, then checked against held-out tests that combine features. Gap between visible and held-out grows ~28 pp per 10× code size; one 2,900-line "compiler" was a hash table memorizing test inputs. A second model reviewing only the implementation and its scores isn't independent verification — the architecture of verification matters, not the instruction to verify.
 
`/goal` is a real execution-control mechanism (objective + lifecycle externalized), not a reminder. It does not guarantee the goal was met: the completion judgment is still a model reading evidence.
 
## `/batch`
 
```text
/batch migrate src/ from JavaScript to TypeScript
```
 
1. Researches the affected code
2. Splits into 5–30 independent units
3. Presents the plan for approval
4. One background subagent per unit, each in its own worktree
5. Each implements, tests and publishes its change — reviewable separately
Requires a git repo or a `WorktreeCreate` hook (outside git: v2.1.281+).
 
- Right when workers have separable file ownership and stable interfaces.
- Worktrees prevent edit collisions, not contradictory contract decisions (semantic coordination). Isolation ≠ dependency management + integrated verification.
- Don't judge by wall-clock alone: total tokens, duplicated work, human interventions, retries, integration conflicts, final correctness. A tightly coupled migration can be faster sequentially.
## Sources
 
- https://code.claude.com/docs/en/scheduled-tasks
- https://code.claude.com/docs/en/goal
- https://code.claude.com/docs/en/commands
- https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex
- https://arxiv.org/abs/2605.21384 — SpecBench
