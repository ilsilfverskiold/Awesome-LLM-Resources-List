# Creating Skills (Claude Code & Codex)
 
Skill → a documented process: a blueprint for how to do something, loaded into the agent's context window. Can be just a prompt, or a folder with references, scripts, tests and evals.
 
A Skill describes what to do and how; an agent definition configures who runs it and with what capabilities. They can be combined.
 
| What you need | Where it belongs |
| --- | --- |
| Product behavior and architectural truth | `docs/` |
| Instructions the agent should always have | `AGENTS.md` / `CLAUDE.md` |
| A validated, repeatable way of carrying out a task | Skill |
| A separate context to investigate or review | Subagent |
| A new external capability | MCP/tool |
| Mechanical checks that must run automatically | Tests, linters and hooks |
 
## When to build one
 
Get the workflow working manually first, then encode it. Don't build a Skill the agent hasn't already executed successfully — you'll get slop.
 
| Step | |
| --- | --- |
| 1 | Encounter a recurring problem |
| 2 | Understand why it happened and establish a working procedure |
| 3 | Encode the reusable procedure in a Skill |
| 4 | Test on fresh tasks with and without the Skill |
| 5 | Keep it only if it improves results |
 
| Weak candidate | Stronger candidate |
| --- | --- |
| Write clean, maintainable code | Reconcile Apple subscription notifications using the existing event writer |
| Remember to test changes | Verify an actual image-generation flow, including failed jobs and state restoration |
| Always make good UI | Run a specific UI review procedure using the app and capture evidence |
| Be careful with databases | Run a validated migration and rollback workflow |
 
| Situation | What a Skill could actually contribute |
| --- | --- |
| Common Apple subscription integration | Your app's specific entitlement rules, unusual failure cases or validated reconciliation workflow — not generic StoreKit knowledge |
| Live Activities and Dynamic Island integration | Current, verified API usage and working examples where documentation is insufficient |
| Your custom statistics API | Meaning of endpoints, how requests must be chained, required identifiers, non-obvious response conventions |
| Internal deployment workflow | Actual order of operations, environment restrictions, rollback process, observed gotchas |
 
- The distinction: knowledge the model already has or can reliably obtain vs knowledge it cannot reconstruct accurately.
- Verification Skills have the strongest evidence: Skills that actually exercise the product (signup/checkout flows in a browser, invoice states, terminal apps) — not "test thoroughly". [unverified: Anthropic internal claim]
- A Skill triggering is not evidence that it works. Test with and without it in fresh sessions.
## Anatomy of a Skill
 
| Part | Why it's there |
| --- | --- |
| description | Helps select the Skill for the right request |
| Objective and required inputs | Defines the specific operation and prerequisites |
| Workflow | Establishes ordering and points to executable checks |
| Gotchas | Captures failures or misconceptions the model may not anticipate — actual discovered knowledge, not speculative warnings |
| References | Detailed knowledge not needed on every invocation |
| Scripts | Repeatable checks without asking the model to reinvent them |
| Output contract | Makes the result auditable rather than just plausible |
 
| Directory | Purpose | Example |
| --- | --- | --- |
| `scripts/` | Code the agent executes to perform an operation | Fetch keyword statistics, handle pagination, validate API responses |
| `tests/` | Test whether that executable code works | Pagination handling, errors, invalid responses, expected transformations |
| `evals/` | Test whether an agent using the Skill behaves correctly | Give a statistics question; check it selects the right workflow and gets the right answer |
| `references/` | Detailed material loaded only when the procedure calls for it | `endpoint-map.md`, `gotchas.md` |
| `assets/` | Templates and static files | `results-template.json` |
| `workflows/` | One file per procedure when the Skill covers several | `investigate-keyword.md`, `generate-content.md` |
 
- Keep `SKILL.md` under 500 lines; move detail into referenced files (verified: Claude Code skills docs).
- Scripts for repeatable mechanics that need deterministic execution, not for every decision (verified: Codex — "Prefer instructions over scripts unless you need deterministic behavior or external tooling").
- Scripts inside a Skill need their own input/output tests.
```text
stats-api/
├── SKILL.md
├── workflows/
│   ├── investigate-keyword.md
│   ├── investigate-trends.md
│   └── generate-content.md
├── references/
│   ├── endpoint-map.md
│   ├── response-contracts.md
│   └── gotchas.md
├── scripts/
│   ├── api_client.py
│   ├── fetch_trends.py
│   └── validate_results.py
├── tests/
│   ├── test_api_client.py
│   ├── test_fetch_trends.py
│   └── fixtures/
│       └── api_responses.json
└── evals/
    ├── evals.json
    └── files/
        └── sample_dataset.json
```
 
## How Skills load
 
| Stage | What loads | |
| --- | --- | --- |
| 1 Discovery | Name, description, path | Small metadata catalogue the model always sees |
| 2 Activation | Full `SKILL.md` | Loads the selected procedure into context |
| 3 Execution | References and scripts | Only when the procedure calls for them |
 
- Once loaded, a Skill body stays in context for the rest of the conversation (verified: Claude Code — one message; identical re-invoke adds a note, not a copy; after compaction the first 5,000 tokens of each invoked skill are re-attached, 25,000 combined).
- The description decides selection. "Use for databases, queries, models or persistence" triggers on nearly any database task; "schema migrations and rollouts" doesn't.
- Claude Code: combined `description` + `when_to_use` listing capped at 1,536 chars (`skillListingMaxDescChars`); whole listing budgeted at 1% of context (`skillListingBudgetFraction`); least-invoked descriptions dropped first when over budget.
- Codex: `[skills] max_context_tokens` — default 2% of context window (8,000 chars if unknown), max 10,000; descriptions shortened first, then skills omitted with a warning.
- Under-trigger (Anthropic, tasks the model thinks it can handle alone [unverified]) vs over-trigger (OpenAI: "front-load the key use case and trigger words").
## Claude Code vs Codex
 
| | Claude Code | Codex |
| --- | --- | --- |
| Project Skills | `.claude/skills/` | `.agents/skills/` (CWD up to repo root) |
| Personal Skills | `~/.claude/skills/` | `~/.agents/skills/` |
| Other locations | enterprise, nested, `--add-dir`, plugins, claude.ai synced | `/etc/codex/skills`, bundled |
| Required | `SKILL.md` with `name`, `description` | `SKILL.md` with `name`, `description` |
| Optional metadata file | — (all in frontmatter) | `agents/openai.yaml` inside the skill folder |
| Can read a SKILL.md at another path when told | Yes | Yes | [unverified]
| Symlinked skill folders | Yes (loaded once even if several locations point at the same target) | Yes (follows the symlink target) |
| Scans `.claude/skills/` | — | Not documented |
| Explicit invoke | `/skill-name args` | `$skill-name`, `/skills` (`@` in ChatGPT) |
| Turn auto-selection off | `disable-model-invocation: true` | `policy.allow_implicit_invocation: false` |
| Per-skill on/off | `/skills` → `skillOverrides` in `.claude/settings.local.json` | `[[skills.config]] path + enabled` in `config.toml` |
| Listing budget | `skillListingBudgetFraction`, `skillListingMaxDescChars` | `[skills] max_context_tokens` |
| Skill creator | `/plugin install skill-creator@claude-plugins-official` → `/skill-creator` | built-in `$skill-creator` |
| Eval runner | `evals/evals.json` + trigger set, with/without baseline, grader, viewer | Not documented — structural validation + iterate through real usage |
 
One Skill, both tools — canonical copy in `.agents/skills/`, symlink from `.claude/skills/`:
 
```text
project/
├── .agents/
│   └── skills/
│       └── inference-engineer/
│           ├── SKILL.md
│           ├── references/
│           └── scripts/
└── .claude/
    └── skills/
        └── inference-engineer → symlink
```
 
FYI: `paths`, `hooks`, `context: fork` are Claude-only frontmatter — a stricter cross-platform uploader may reject them. Common format first; add Claude-specific fields only when needed; test in both.
 
## Claude Code
 
### Creating one
 
- Ask Claude to create it under `.claude/skills/`, or `/skill-creator` after:
```text
/plugin install skill-creator@claude-plugins-official
```
 
- If `Marketplace "claude-plugins-official" not found` → `/plugin marketplace add anthropics/claude-plugins-official` first; `/reload-plugins` if asked. Also installable from the plugin interface in the VS Code extension.
- Skill Creator proposes test prompts, should/shouldn't-trigger examples and expected outcomes — you review them where correct behavior depends on knowledge only you have.
### Skill file
 
`.claude/skills/stats-api/SKILL.md`
 
```markdown
---
name: stats-api
description: >
  Investigate statistics using the internal API, including
  following relationships between summary and detail results.
when_to_use: >
  Use for investigations requiring statistics to be traced
  through multiple API responses.
---
Investigate the API issue supplied in $ARGUMENTS.
Read relevant code and documentation.
Return findings with evidence.
```
 
### Frontmatter fields
 
| Field | What it does | Example |
| --- | --- | --- |
| `name` | Skill name and command (`/stats-api`) | `name: stats-api` |
| `description` | What it does; drives selection | `description: Review schema migrations…` |
| `when_to_use` | Extra triggering text, appended to description in the catalogue; not a second rule engine | `when_to_use: Use for investigations requiring…` |
| `argument-hint` | Autocomplete hint; doesn't validate types | `argument-hint: "[workflow] [topic] [period]"` |
| `arguments` | Named args → `$workflow`, `$topic`; omitted → empty string (omitted `$2` stays literal) | `arguments: [workflow, topic, period]` |
| `disable-model-invocation` | `true` → only you can invoke it | `disable-model-invocation: true` |
| `user-invocable` | `true` (default) visible as a command; `false` hides it from you, Claude can still pick it | `user-invocable: false` |
| `allowed-tools` | Preapproves tools for the invoking turn; doesn't restrict others; applied even in untrusted folders and `-p` | `allowed-tools: [Read, Grep, Glob]` · `Bash(git status *)` |
| `disallowed-tools` | Removes tools while the Skill is active (can't remove `EndConversation`) | `disallowed-tools: [Write, Edit]` |
| `model` | Model for the invocation / forked worker | `model: sonnet` |
| `effort` | Reasoning effort; levels depend on model | `effort: high` |
| `context` | `fork` → fresh subagent context (not your history) | `context: fork` |
| `agent` | Which agent runs the fork: `Explore`, `Plan`, `general-purpose` (default) or custom; `Explore` sees SKILL.md + its own prompt, no CLAUDE.md | `agent: Explore` |
| `background` | `true` (default v2.1.218+) fork runs in background; `false` main agent waits | `background: false` |
| `paths` | Auto-load only when working on matching files; same format as path rules | `paths: ["src/payments/**/*.ts"]` |
| `hooks` | Hooks registered on invoke; run for the rest of the session | see below |
 
- `allowed-tools`/`disallowed-tools` effects clear on your next message; the Skill text stays in context.
- Forbidding `Write`/`Edit` isn't read-only if `Bash` is still unrestricted.
- Substitutions: `$ARGUMENTS`, `$ARGUMENTS[0]`, `$0`, `$name`, `${CLAUDE_SKILL_DIR}`, `${CLAUDE_PROJECT_DIR}`, `${CLAUDE_SESSION_ID}`, `${CLAUDE_EFFORT}`.
### Invoking with arguments
 
```markdown
---
name: investigate-stat
description: Investigate a statistics record through the internal API
argument-hint: "[stat-id]"
arguments:
  - stat_id
disable-model-invocation: true
user-invocable: true
---
Investigate statistics record $stat_id.
```
 
```text
/investigate-stat example-123
/stats-api keyword "artificial intelligence" 30d
/stats-api trending
```
 
### Tools
 
| Tool | What it allows | Example |
| --- | --- | --- |
| `Read` | Open a file | Read `api-workflow.md` |
| `Grep` | Search file contents | Find references to `trendId` |
| `Glob` | Find files by name pattern | All `*.tsx` |
| `Write` | Create/overwrite a file | Create a report |
| `Edit` | Change an existing file | Fix an API handler |
| `Bash` | Run shell commands | Tests, a Python script |
| `Agent` | Delegate to another agent | Ask a reviewer to investigate |
 
### Running in a subagent
 
```markdown
---
name: review-api
description: Independently investigate an API issue
model: sonnet
effort: high
context: fork
agent: Explore
background: false
---
```
 
Without `context: fork` the Skill runs in the current context; `agent` only matters with a fork. `background: false` → the main agent waits for the subagent.
 
### Visibility
 
`/skills` → select → Space cycles state. Saved to `.claude/settings.local.json` (plugin skills not affected):
 
```json
{
  "skillOverrides": {
    "review-schema-migration": "user-invocable-only"
  }
}
```
 
| State | Claude sees description | You can invoke |
| --- | --- | --- |
| `on` | Yes | Yes |
| `name-only` | Name only | Yes |
| `user-invocable-only` (shown as `user-only`) | No | Yes |
| `off` | No | No |
 
### Hooks in a Skill
 
| Hook location | Active from | Stops |
| --- | --- | --- |
| `.claude/settings.json` | Session start | Session end / config change |
| Skill `SKILL.md` | Skill invoked | Session end, unless removed (`once: true` removes after first success) |
| Subagent config | Subagent start | Subagent finish |
 
```markdown
---
name: checked-reading
description: Read project files with a preliminary check
hooks:
  PreToolUse:
    - matcher: "Read"
      hooks:
        - type: command
          command: "${CLAUDE_PROJECT_DIR}/scripts/before-read.py"
---
```
 
- Script gets the tool call as JSON on stdin (`tool_input.file_path` for Read). Exit `2` blocks; `0` continues.
- Matchers: `Read`, `Glob`, `Grep`, `Bash`, `Edit`, `Write`, `Agent`, …
- Doesn't fire for `@file` references [unverified] or `cat` via Bash. Not an OS-level boundary.
- Use only for behavior that matters after a specific workflow starts (e.g. collecting diagnostics); a hook activated mid-session keeps running for unrelated reads.
### Evaluating
 
| Test | Question | How |
| --- | --- | --- |
| Structural | Correctly packaged? | Frontmatter, file structure, scripts |
| Discoverability | Selected for the right requests, not the wrong ones? | Natural-language queries → check invocations |
| Effectiveness | Better with it than without? | Same task, fresh sessions, with vs without |
| Regression | Still fine after an edit? | Rerun earlier cases, not just the one you fixed |
 
Files:
 
`evals/evals.json` — task quality (verified: `references/schemas.md` in skill-creator; its SKILL.md says `assertions` — inconsistency):
 
```json
{
  "skill_name": "review-schema-migration",
  "evals": [
    {
      "id": 1,
      "prompt": "Review the migration removing the legacy customer ID column.",
      "expected_output": "A migration review with findings supported by code and test evidence.",
      "files": [],
      "expectations": [
        "Identifies potential breaking consumers of the removed column",
        "Reports the verification commands that actually ran",
        "Does not report successful verification without evidence"
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
| `evals[].files` | Input files for the test |
| `evals[].expectations` | Statements the grader checks |
 
Trigger set — discoverability; separate file, saved in the eval workspace by the description-optimization workflow:
 
```json
[
  { "query": "Review the migration that drops the old customer ID column before deployment.", "should_trigger": true },
  { "query": "Our customer search SQL query is slow. Find the bottleneck.", "should_trigger": false }
]
```
 
| Request | Expected |
| --- | --- |
| "Add a database migration to rename the customer ID column." | Trigger |
| "Review this migration before deployment." | Trigger |
| "Optimize a slow database query." | Do not trigger |
| "Explain how foreign keys work." | Do not trigger |
 
Run:
 
1. Write `SKILL.md` + supporting files.
2. Save 2–3 test cases to `evals/evals.json` (or let Skill Creator propose them).
3. Ask: `Evaluate the stats-api Skill using Skill Creator. Start from its existing evals/evals.json. Review the cases with me, then run fresh evaluations with and without the Skill. Grade the results; show the comparison, failures and token costs. Separately evaluate whether the description triggers for the right requests.`
4. Skill Creator creates the workspace incrementally, runs A (with Skill) and B (without) as fresh subagents, grades (another agent, executable checks, or you), writes `benchmark.json`/`benchmark.md`, opens a review viewer. Revised Skill → compares new vs previous version (blind A/B).
```text
review-schema-migration/
├── SKILL.md
└── evals/
    └── evals.json
review-schema-migration-workspace/
└── iteration-1/
    ├── eval-0/
    │   ├── with_skill/
    │   │   ├── outputs/
    │   │   └── grading.json
    │   └── without_skill/
    │       ├── outputs/
    │       └── grading.json
    ├── benchmark.json
    └── benchmark.md
```
 
| Job | Who |
| --- | --- |
| Define what the Skill should do and what correct looks like | You, assisted by the agent |
| Generate cases and run sessions | Skill Creator |
| Decide whether the tests measure the right thing | You + independent executable checks |
 
- Compare: correctness, verification performed, files changed, unnecessary steps, token use, human corrections, execution time.
- Subjective work (UI) → human side-by-side with a rubric; objective work → executable checks.
- One passing case proves nothing. After an edit, rerun the related cases too.
- Description tuning: generates should/shouldn't-trigger prompts, measures hit rate, proposes edits; multiple runs per query. (~20 queries, 60/40 split, 3 trials — [unverified])
## Codex
 
### Creating one
 
```text
$skill-creator
```
 
- Asks what the skill does, when it triggers, instruction-only (default) or with scripts; generates `agents/openai.yaml`; validates the package (`quick_validate.py` — naming, frontmatter, required fields [unverified: not on current docs page]). Record & Replay can draft one from a demonstration.
- No documented `evals.json`-style runner: structural validation + iterate through real use. Paired with/without runs are something you set up yourself.
### Skill folder
 
```text
.agents/skills/review-schema-migration/
├── SKILL.md              # Required: name, description, instructions
├── agents/
│   └── openai.yaml       # Optional: display, invocation policy, tool dependencies
├── scripts/              # Optional
├── references/
└── assets/
```
 
`SKILL.md`
 
```markdown
---
name: review-schema-migration
description: Review schema migrations for breaking changes
  and deployment risks. Use when creating or reviewing
  a database schema migration.
---
# Instructions
1. Do this first
2. Do this after
3. Do this last
```
 
### `agents/openai.yaml`
 
| Field | What it does |
| --- | --- |
| `interface.display_name` | Display name |
| `interface.short_description` | Short display description |
| `interface.icon_small` / `icon_large` | Icon files |
| `interface.brand_color` | Display color |
| `interface.default_prompt` | Suggested prompt |
| `policy.allow_implicit_invocation` | Auto-select from the user's task (default `true`); `false` → only `$skill-name` |
| `dependencies.tools` | Tool dependencies incl. MCP connections |
 
```yaml
interface:
  display_name: "Stats API Guide"
  short_description: "Investigate statistics using the internal API"
  icon_small: "./assets/stats-small.svg"
  icon_large: "./assets/stats-large.png"
  brand_color: "#3B82F6"
  default_prompt: "Use this skill to investigate a statistics question"
 
policy:
  allow_implicit_invocation: true
 
dependencies:
  tools:
    - type: "mcp"
      value: "internalStats"
      description: "Internal statistics MCP server"
      transport: "streamable_http"
      url: "https://example.com/mcp"
```
 
### `config.toml`
 
`~/.codex/config.toml` or `.codex/config.toml`
 
```toml
[skills]
max_context_tokens = 6000
 
[[skills.config]]
path = "/Users/me/projects/my-app/.agents/skills/old-api/SKILL.md"
enabled = false
```
 
- `max_context_tokens`: budget for the initial skills list only; full `SKILL.md` still read on selection.
- `[[skills.config]]`: absolute `path` + `enabled`.
## MCP vs Skill with scripts
 
You don't need an MCP server for a Skill to call an API — a script can.
 
| Option | Flow | Notes |
| --- | --- | --- |
| A — Skill + scripts | Load `SKILL.md` → run `scripts/stats_cli.py` → HTTP API | No MCP. Skill says when to run the CLI and how to read the output |
| B — MCP server | Agent picks `get_keyword_stats` tool → server validates, calls API | Typed interface; still needs instructions for multi-endpoint workflows |
| C — Skill + MCP | Skill picks the workflow → agent calls MCP tools in sequence | Procedure separated from the authenticated tool interface |
 
- MCP wins for typed inputs, controlled permissions, stateful connections, hidden auth.
- "MCP loads huge tool schemas" is outdated where deferred tool discovery is on; overhead depends on client/server/settings.
- Practitioner accounts, not benchmarks.
```bash
python scripts/stats_cli.py keyword --term "artificial intelligence" --period 30d
python scripts/stats_cli.py trending --period 7d
python scripts/stats_cli.py sources --keyword-id 123
```
 
## Failure modes
 
| Problem | What happens | What helps |
| --- | --- | --- |
| Too many Skills | Overlapping descriptions → poor selection; metadata shortened or skills dropped to fit budget | Narrow descriptions; turn off unused Skills |
| Outdated instructions | Skill built for an older model/architecture becomes useless or harmful | Re-evaluate after model or project changes |
| Context interference | Related-but-wrong guidance (e.g. older API version) overrides the model's correct default | Version-specific facts in the Skill; test on the real repo |
| Evaluation contamination | Agents reading eval material (UiPath audit: 2.4% of runs [unverified]) | Separate graders and test data from the agent's reach |
| Security | Third-party Skills = scripts, dependencies, hooks, instructions — installing software, not copying docs | Review before installing |
| Context consumption | Metadata always loaded; body stays once loaded | Short `SKILL.md`, detail in references |
| Self-generated Skills | No benefit on average (SkillsBench −1.3 pp; Codex/GPT-5.2 −5.6 pp) | Evaluate an AI-written Skill, don't assume it helps |
 
For a shared engineering Skill: adoption · discoverability · effectiveness · currency · ownership · scripts/external access.
 
## Research
 
| Paper | Setup | Result |
| --- | --- | --- |
| SWE-Skills-Bench (arXiv 2603.15401, Mar 2026, pre-print) | 49 public SWE Skills, ~565 tasks, Claude Code + Haiku 4.5 | 39/49 no change (24 both 100%, 15 both failing), 7 up, 3 down; avg +1.2%; tokens +10.5% avg, range −77.6% to +450.8% |
| SkillsBench (arXiv 2602.12670, Feb 2026) | 84 tasks, 11 domains, 7 agent configs (Claude Code, Codex CLI, Gemini CLI) | Curated Skills +16.2 pp avg; 16 tasks negative; self-generated −1.3 pp |
| In the Wild (arXiv 2604.04323, Apr 2026) | Opus 4.6, Claude Code v2.1.19, 34,198-skill pool | Loads all curated Skills in 49% of runs, 31% with distractors; pass 55.4% → 51.2% → 43.5%; Recall@5 65.5% |
 
SWE-Skills-Bench per Skill (Table 2):
 
| Skill | Pass rate | Tokens |
| --- | --- | --- |
| risk-metrics-calculation | +30.0 pp (70→100%) | −34.8% |
| gitlab-ci-patterns | +14.3 pp | +58.6% |
| prompt-engineering-patterns | +10.0 pp | +46.4% |
| similarity-search-patterns | +10.0 pp | −32.4% |
| distributed-tracing | +7.7 pp | −30.4% |
| tdd-workflow | +7.1 pp (21.4→28.6%) | +78.6% |
| istio-traffic-management | +7.1 pp | −22.0% |
| springboot-tdd | −10.0 pp | |
| linkerd-patterns | −9.1 pp | |
| django-patterns | −9.1 pp | |
| service-mesh-observability | 0 | +450.8% |
 
SkillsBench by domain (Table 4, curated Skills, pp over baseline): Healthcare 51.9 · Manufacturing 41.9 · Cybersecurity 23.2 · Natural science 21.9 · Energy 17.9 · Office work 17.8 · Finance 15.1 · Media/content 13.9 · Robotics 7 · Mathematics 6 · Software engineering 4.5.
 
Takeaway: Skills help where they supply missing domain knowledge, validated procedures, tool integrations or real verification; redundant or harmful when they tell an already-capable model how to write ordinary code.
 
## Sources
 
- https://code.claude.com/docs/en/skills
- https://code.claude.com/docs/en/sub-agents
- https://code.claude.com/docs/en/hooks
- https://learn.chatgpt.com/docs/build-skills.md
- https://github.com/anthropics/claude-plugins-official (skill-creator)
- https://arxiv.org/abs/2603.15401 — SWE-Skills-Bench
- https://arxiv.org/abs/2602.12670 — SkillsBench
- https://arxiv.org/html/2604.04323v1 — How Well Do Agentic Skills Work in the Wild?
