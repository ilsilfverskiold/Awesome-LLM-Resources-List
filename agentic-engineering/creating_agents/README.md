# Creating Subagents (Codex & Claude Code)
 
Subagent → a separate model context with a focused task. Own instructions; gets a delegation prompt plus selected Skills/project context; returns a result. Does not see the parent's conversation.
 
| Approach | What you do | Example |
| --- | --- | --- |
| Use a built-in agent | Ask the main agent to delegate, or let it decide | "Investigate the payment flow using a read-only subagent." |
| Create a reusable custom agent | Save a definition: role, model, instructions, capabilities | `ux-reviewer` |
| Delegate in parallel | Ask for several separate investigations or bounded tasks | "Have three agents independently investigate these possible causes." |
 
Both tools: describe the agent in ordinary language and let the assistant write the file, or write it yourself.
 
## Claude Code vs Codex
 
| | Claude Code | Codex |
| --- | --- | --- |
| Project-level definitions | `.claude/agents/*.md` | `.codex/agents/*.toml` |
| User-level definitions | `~/.claude/agents/` | `~/.codex/agents/` |
| Format | Markdown + YAML frontmatter | TOML |
| Required fields | `name`, `description` | `name`, `description`, `developer_instructions` |
| Built-in agents | Explore, Plan, general-purpose, helpers (`claude`, `statusline-setup`, `claude-code-guide`) | `default`, `worker`, `explorer` |
| Instructions | Markdown body | `developer_instructions` |
| Skills | `skills` list | `skills.config` |
| MCP servers | `mcpServers` | Inherited from parent; not settable per agent [corrected: was `mcp_servers`] |
| Tools | allow/deny lists | `features.*` keys to switch capabilities off; sandbox/approval come from the parent session [corrected: was "session tool and sandbox configuration"] |
| File isolation | `isolation: worktree` | Managed separately (no per-agent key) |
| Agent command | `/agents` — prints guidance; wizard removed in v2.1.198 | `/subagents` — switch between running agent threads [corrected: was `/agent`] |
 
- Not a one-to-one schema: don't paste Claude `tools`/`skills` entries into Codex TOML.
- `/import codex` (Claude Code) imports Codex config incl. subagents and Skills — one-time, not sync.
- Skills: Codex reads `.agents/skills/`; Claude Code reads `.claude/skills/`. Share with one canonical copy in `.agents/skills/` and a symlink in `.claude/skills/` (Claude follows symlinked Skill dirs).
## Claude Code
 
### Where they go
 
```text
your-project/
└── .claude/
    ├── agents/
    │   └── ux-reviewer.md        # project-level subagent
    └── settings.json             # e.g. CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS
~/.claude/
└── agents/                       # user-level, every project
```
 
### Built-in agents
 
| Agent | Purpose | Detail |
| --- | --- | --- |
| Explore | Search and understand the codebase | Read-only; skips CLAUDE.md and Git status |
| Plan | Investigate during Plan Mode | Read-only; skips CLAUDE.md and Git status |
| General-purpose | Multi-step work incl. editing | Full tool set; normal project context |
 
Different workers get different starting contexts on purpose.
 
### Custom agent file
 
`.claude/agents/ux-reviewer.md`
 
```markdown
---
name: ux-reviewer
description: Review implemented UI for product and UX problems
model: sonnet
effort: high
tools: Read, Grep, Glob
skills:
  - verify-product-ui
maxTurns: 12
---
Review the implemented UI against the product model
and the approved requirements.
Find concrete behavior or UX problems.
Return file references and describe the impact.
Do not edit the implementation.
```
 
### Configuration fields
 
Only `name` and `description` are required (frontmatter reference, 27 Oct 2026).
 
| Field | Purpose | When it's useful | Example |
| --- | --- | --- | --- |
| `name` | Identifies the agent | Required; max 256 chars, no `:` | `name: ux-reviewer` |
| `description` | When Claude should delegate to it | Required; >15,000 tokens combined → startup warning | `description: Reviews code. Use proactively after code changes.` |
| `model` | Model to use | Cost/capability per worker | `model: haiku` · `inherit` · `claude-opus-5-5` |
| `effort` | Reasoning effort | Hard investigations/reviews; `low`–`max` | `effort: high` |
| `tools` | Allowlist | Read-only reviewer vs editing worker; inherits all if omitted | `tools: Read, Grep, Glob` |
| `disallowedTools` | Denylist | Deny writes etc.; `Bash(git push *)` removes all of Bash | `disallowedTools: Write, Edit` |
| `permissionMode` | Approval behavior | `default`, `acceptEdits`, `auto`, `dontAsk`, `bypassPermissions`, `plan`; ignored for plugin agents | `permissionMode: plan` |
| `skills` | Preload Skills | Validated workflow; full content injected, max 32 | `skills: [api-conventions]` |
| `mcpServers` | Extra external tools | Browser, docs, issue trackers; ignored for plugin agents | `mcpServers: [github]` |
| `hooks` | Lifecycle/tool checks | Validate or block operations; ignored for plugin agents | see below |
| `memory` | Persistent memory | Reuse learnings; `user`, `project`, `local` | `memory: project` |
| `maxTurns` | Turn cap | Bound a task; partial output is resumable (v2.1.246+) | `maxTurns: 12` |
| `background` | Non-blocking | Already default from main conversation; `true` keeps it there even when Claude wants to wait | `background: true` |
| `omitClaudeMd` | Skip CLAUDE.md | Self-contained tasks; managed policy still loads (v2.1.271+) | `omitClaudeMd: true` |
| `isolation` | Separate working files | Independent code-writing agents | `isolation: worktree` |
| `color` | UI color | Tell concurrent agents apart | `color: cyan` |
| `initialPrompt` | First turn as main-session agent | `claude --agent <name>` | `initialPrompt: Review the open pull requests` |
| `experimental` | Prompt cache TTL | `5m` or `1h` (v2.1.248+) | `experimental: {cacheTtl: 1h}` |
 
Reference: [frontmatter reference](https://code.claude.com/docs/en/sub-agents#supported-frontmatter-fields)
 
Hook example — script runs before each Bash command; exit code 2 blocks it:
 
```markdown
---
name: db-reader
description: Execute read-only database queries
tools: Bash
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate-readonly-query.sh"
---
```
 
### What can go inside `tools`
 
A role description doesn't grant a capability; only `tools` does.
 
| Type | Examples |
| --- | --- |
| Explore code | `Read`, `Grep`, `Glob`, `LSP` |
| Modify code | `Edit`, `Write`, `NotebookEdit` |
| Run commands | `Bash`, `PowerShell` |
| Look up information | `WebFetch`, `WebSearch` |
| Load procedures | `Skill` |
| Delegate | `Agent` |
| Find tool definitions | `ToolSearch` |
| External | MCP tools (browser, docs, …) |
 
### Can an agent call other agents?
 
```yaml
tools: Read, Grep, Glob, Agent
```
 
Nesting: 3 layers below main by default; `CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH` changes it (`1` = off).
 
Main-session coordinator (`claude --agent …`) — restrict which agents it may spawn:
 
```yaml
tools: Agent(researcher, reviewer), Read, Bash
```
 
Names must match existing definitions. The list in parentheses is ignored inside a subagent definition.
 
### `/subtask` vs `/fork` vs normal subagent
 
| | Starts with | Comes back |
| --- | --- | --- |
| Normal subagent | Fresh context + delegated task | Result |
| `/subtask` | Copy of the whole conversation | Result, in this conversation |
| `/fork` | Copy of the conversation as a separate background session | Separate session you can return to or message |
 
- The fork receives the history; the main conversation only receives the result.
- `/subtask` shares the parent's prompt cache → cheaper than a fresh worker for same-context tasks.
- Agent view off → `/subtask` unavailable, `/fork` spawns the forked subagent instead.
```text
/subtask Investigate whether our current subscription
implementation correctly handles duplicate renewals.
Report findings without modifying files.
```
 
### Agent teams (experimental)
 
Subagent → reports to parent. Team → workers message each other. Experimental and off by default (v2.1.32+).
 
```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```
 
Then ask Claude to create a team (not just parallel subagents). Limitations: no resume of in-process teammates, task status can lag, slow shutdown, one team per session, no nested teams.
 
### Ultracode (dynamic workflows)
 
Claude writes a JavaScript script that decides which agents run, when, and how results combine; runs in a separate runtime, intermediate results stay out of Claude's context.
 
- Trigger: keyword `ultracode` in a prompt, or `/effort ultracode` for the session.
- `agent()` one subagent · `parallel()` several at once · `pipeline()` a list · `phase()`, `log()`.
- Handles loops, branching, verification rounds, retries in code.
- Save a run from `/workflows` → reusable command.
- Limits: 16 concurrent agents (default), 1,000 per run.
## Codex
 
### Where they go
 
```text
your-project/
└── .codex/
    └── agents/
        └── ux-reviewer.toml      # project-level
~/.codex/
└── agents/                       # user-level
```
 
### Built-in agents
 
`default`, `worker`, `explorer`. Delegation is on by default locally; ask directly. A custom agent with a built-in name wins.
 
### Custom agent file
 
`.codex/agents/ux-reviewer.toml`
 
```toml
name = "ux_reviewer"
description = "Review implemented UI for product and UX problems"
model = "gpt-6-luna"
model_reasoning_effort = "high"
developer_instructions = """
Review the implemented UI against the product
model and approved requirements.
Use the verify-product-ui Skill when applicable.
Return concrete findings with file references.
Do not modify code.
"""
 
[[skills.config]]
path = "/absolute/path/to/.agents/skills/verify-product-ui/SKILL.md"
enabled = true
```
 
[corrected: removed `sandbox_mode = "read-only"` — a custom agent file can't create a read-only sandbox; set sandbox/approval on the parent session]
 
- Required: `name`, `description`, `developer_instructions`. Model: `model`, `model_reasoning_effort`, `model_reasoning_summary`, `model_verbosity`.
- `[[skills.config]]`: `path` + `enabled`; skills inherit from parent, documented use is `enabled = false`.
- Ignored in an agent file: `sandbox_mode`, `approval_policy`, `model_provider`, `mcp_servers`.
### `/subagents`
 
Switch between running agent threads in the CLI. Not a creation wizard.
 
## What benefits from separate context or parallel execution
 
| Work | Setup | Why |
| --- | --- | --- |
| Small change, known patterns | Main agent | Delegation adds overhead |
| Large codebase investigation | One subagent | Keeps intermediate output isolated |
| Several plausible causes | Multiple read-only subagents | Independent investigations can be compared |
| Reviewing an implementation | Independent reviewer | Doesn't inherit the implementer's reasoning |
| Several independent code changes | Separate worktrees | No overwriting each other's edits |
| Connected architectural changes | Coordinator with selective delegation | Someone owns the decisions and dependencies |
| Repetitive large migration | `/batch` (Claude Code) | 5–30 bounded units, one subagent each in its own worktree |
 
## Failure modes
 
| Problem | Why | What helps |
| --- | --- | --- |
| Token costs increase quickly | Every agent does its own model/tool work and may reload context | Fewer, narrower workers; don't load all project docs into every agent |
| Worker misses information | Separate context ≠ the whole discussion | Pass the contract, constraints, source docs |
| Duplicate work | Overlapping tasks / same problem assigned twice | Clear assignments and outputs |
| Changes fail together | Connected code changed on incompatible assumptions | Shared interfaces, integration tests, communication |
| Reviewer agrees with wrong code | Independent context ≠ independent judgment | Separate behavioral requirements, evidence, real verification |
| Supervision overwhelms you | More parallel work → more reviews, integrations, decisions | Limit concurrency to what you can check |
 
Token cost — ten subagents, `omitClaudeMd`:
 
| | |
| --- | --- |
| 226,340 | tokens, default loading |
| 26,480 | after project context omitted (−88%) |
 
([source](https://www.reddit.com/r/LLMDevs/comments/1wighvz/found_a_small_way_to_save_some_claude_code/)) [unverified]
 
## Research example
 
Bug: generated media attached to the wrong sketch. Tracing it = twenty files, IDs across services, several possible causes. A subagent does the exploration and returns file locations and evidence.
 
## Worktrees
 
Separate code changes, not separate thinking. A subagent shares the project's working files; own context ≠ isolated edits. A Git worktree = separate checkout and branch of the same repository.
 
| | Gives you |
| --- | --- |
| Subagent | separate model context |
| Worktree | separate filesystem/git state |
 
- Claude Code: `isolation: worktree`. Temporary worktree branched from the default branch (or `HEAD` with `worktree.baseRef: "head"`); removed if no changes, kept if changes; edits/git commands against the main checkout are blocked.
- Multiple independent code changes → separate worktrees → no overwriting.
- Parallel implementation — set shared interfaces and ownership first:
```text
Implement the approved profile feature using two parallel workers. One owns the backend, the other owns the frontend. Use the existing API contract. Keep edits isolated, report the changes and tests from each worker, then integrate and verify the full flow.
```
 
## Sources
 
- [Claude Code: subagents](https://code.claude.com/docs/en/sub-agents)
- [Claude Code: commands](https://code.claude.com/docs/en/commands)
- [Claude Code: agent teams](https://code.claude.com/docs/en/agent-teams)
- [Claude Code: dynamic workflows](https://code.claude.com/docs/en/workflows)
- [Claude Code: worktrees](https://code.claude.com/docs/en/worktrees)
- [Claude Code: skills](https://code.claude.com/docs/en/skills)
- [Codex: subagents](https://developers.openai.com/codex/subagents)
- [Codex: skills](https://developers.openai.com/codex/skills)
- [Reddit: omitClaudeMd experiment](https://www.reddit.com/r/LLMDevs/comments/1wighvz/found_a_small_way_to_save_some_claude_code/) — unverified
