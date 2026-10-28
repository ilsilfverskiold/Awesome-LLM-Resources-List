# Evaluating Skills (Claude Code)
 
A Skill triggering is not evidence that it works. The test is fresh-session runs with and without the Skill. Claude Code's `skill-creator` plugin does this since March 2026 (Anthropic, "Improving skill-creator", 3 Mar 2026). Codex has no equivalent runner.
 
```text
/plugin install skill-creator@claude-plugins-official
```
 
If `Marketplace "claude-plugins-official" not found` → `/plugin marketplace add anthropics/claude-plugins-official` first.
 
## What you're testing
 
| Test | Question | How |
| --- | --- | --- |
| Structural | Correctly packaged? | Frontmatter, file structure, scripts (`scripts/quick_validate.py` in skill-creator) |
| Discoverability | Picked for the right requests, not the wrong ones? | Trigger set → run → check invocations |
| Effectiveness | Better with it than without? | Same task, fresh sessions, with vs without |
| Regression | Still fine after an edit? | Rerun all earlier cases, not just the one you fixed |
 
Effectiveness is the one that matters: a Skill can trigger correctly, run its instructions, and still produce worse work.
 
## Step by step
 
```text
Evaluate the stats-api Skill using Skill Creator. Review the cases and expected behavior with me, then run fresh evaluations with and without the Skill. Grade the actual results, and show the output comparison, failures and token costs.
```
 
| Step | What happens | Files |
| --- | --- | --- |
| 1 | Proposes 2–3 realistic test prompts, asks you to confirm | `stats-api/evals/evals.json` — prompts + `expected_output` only; inputs a test needs go in `evals/files/` |
| 2 | Spawns two subagents per case at the same time: with Skill, without Skill (or the old version, snapshotted, when improving an existing Skill) | `stats-api-workspace/iteration-1/<case>/with_skill/outputs/`, `without_skill/outputs/` — created as it goes |
| 3 | While runs go, drafts the expectations and explains them | `expectations[]` added to `evals.json` |
| 4 | Saves tokens + duration per run; grader agent (or a script) checks each expectation; aggregates | `timing.json`, `grading.json` per run; `benchmark.json` / `benchmark.md` — pass rate, time, tokens, mean ± stddev, delta |
| 5 | Opens a viewer: Outputs tab (click through, leave feedback), Benchmark tab | `feedback.json` |
| 6 | Rewrites the Skill from your feedback, reruns everything, shows last round next to new | `iteration-2/` … until you're happy, feedback is empty, or no progress |
| 7 | Description tuning — separate, after the Skill is good | trigger set in the workspace (see below) |
 
```text
stats-api/                              # the Skill — what the agent loads
└── evals/
    ├── evals.json                      # prompts + expected_output + expectations
    └── files/                          # input files a test points to
 
stats-api-workspace/                    # sibling folder, created by Skill Creator
├── trigger-eval.json                   # query + should_trigger (step 7)
├── feedback.json                       # your viewer comments (step 5)
└── iteration-1/
    ├── <case>/
    │   ├── eval_metadata.json
    │   ├── with_skill/    { outputs/, timing.json, grading.json }
    │   └── without_skill/ { outputs/, timing.json, grading.json }
    ├── benchmark.json
    └── benchmark.md
```
 
## `evals/evals.json`
 
```json
{
  "skill_name": "stats-api",
  "evals": [
    {
      "id": 1,
      "prompt": "Investigate why keyword traffic for 'renewable energy' dropped last month.",
      "expected_output": "A traced investigation across the summary and detail endpoints, with the records and evidence used.",
      "files": [],
      "expectations": [
        "Follows the summary endpoint with the matching detail endpoints",
        "Reports the records and evidence used",
        "Does not report a cause without evidence from the API responses"
      ]
    }
  ]
}
```
 
| Field | Meaning |
| --- | --- |
| `skill_name` | Skill under test |
| `evals[].id` | Test id |
| `evals[].prompt` | Task given to the agent |
| `evals[].expected_output` | What success looks like |
| `evals[].files` | Input files, relative to the skill root |
| `evals[].expectations` | Statements the grader checks — objectively verifiable; skip for subjective work (UI, writing) and use human side-by-side review with a rubric |
 
FYI: `references/schemas.md` says `expectations`; skill-creator's SKILL.md says `assertions`. The schema is what the grader reads.
 
## Trigger set (discoverability)
 
20 queries, 8–10 should-trigger, 8–10 should-not. The useful negatives are near-misses that share keywords but need something else. You review them in an HTML page, export, Skill Creator saves the set to the workspace.
 
```json
[
  { "query": "Investigate why keyword traffic for 'renewable energy' dropped last month.", "should_trigger": true },
  { "query": "Fix the failing unit test in api_client.py.", "should_trigger": false }
]
```
 
`should_trigger` is the expected result, not an instruction the agent sees.
 
Optimization loop (`scripts.run_loop`): 60% train / 40% held-out, each query run 3× for a reliable trigger rate, Claude proposes a new description from the failures, up to 5 rounds, winner picked by held-out score. You apply `best_description` to `SKILL.md`.
 
- Claude only consults a Skill for tasks it can't easily do alone — one-step requests may not trigger even with a perfect description. Skill Creator calls this under-triggering and writes descriptions "a little bit pushy". OpenAI warns about the opposite: over-triggering from broad descriptions.
## Reading the results
 
- Compare: correctness, verification performed, files changed, unnecessary steps, token use, human corrections, execution time.
- A strong Skill contributes something the model would not otherwise do correctly or reliably.
- One passing case proves nothing — it may be a favorable example.
- After an edit, rerun the related cases too, not only the one that failed.
- If every test run independently wrote the same helper script, bundle it in `scripts/`.
- Optional blind comparison: two outputs to an independent agent without saying which is which (`agents/comparator.md`).
## Who does what
 
| Job | Who |
| --- | --- |
| Define what the Skill should do and what correct looks like | You, assisted by the agent |
| Generate cases, run sessions, grade, aggregate | Skill Creator |
| Decide whether the tests measure the right thing | You, plus independent executable checks |
 
## Sources
 
- https://code.claude.com/docs/en/skills
- https://claude.com/blog/improving-skill-creator-test-measure-and-refine-agent-skills
- https://github.com/anthropics/claude-plugins-official/tree/main/plugins/skill-creator
