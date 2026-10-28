# Agentic Engineering (Claude Code & Codex)
 
Reference files for working with coding agents — terse lookup docs, not essays. Each one covers Claude Code and Codex side by side with the config files, fields, tests and sources. Companion to the article series on Towards Data Science.
 
| Level | Topic | File | What's in it |
| --- | --- | --- | --- |
| 1 | Configs & permissions | [configs/README.md](configs/README.md) | Permission modes, allow/ask/deny, starter `settings.json` and `config.toml`, sandboxing, how to test that the agent can't leave the project |
| 2 | `AGENTS.md` / `CLAUDE.md` | [AGENTS/README.md](AGENTS/README.md) | What goes in the always-loaded instructions file, what developers actually put there, examples from public repos |
| 4 | Subagents | [creating_agents/README.md](creating_agents/README.md) | Where agent files go, built-in agents, every frontmatter/TOML field, tools, nesting, `/subtask` vs `/fork`, agent teams, Ultracode, worktrees, failure modes |
| 5 | Skills | [Skills/README.md](Skills/README.md) | When to build one, anatomy, how Skills load, every frontmatter field, arguments, tools, hooks, visibility, Codex `agents/openai.yaml` and `config.toml`, MCP vs scripts, failure modes, research |
| 5.1 | Evaluating Skills | [Skills/evaluating-skills.md](Skills/evaluating-skills.md) | `skill-creator` step by step: `evals/evals.json`, the workspace it creates, grading, benchmark, trigger set and description tuning |
| 5.2 | `/loop`, `/goal`, `/batch` | [Skills/bundled-skills.md](Skills/bundled-skills.md) | Scheduling in-session vs cloud vs Desktop, `loop.md`, goals in Codex and Claude Code, the goal evaluator's limits, reward hacking, parallel `/batch` |
| 5.3 | Community Skills | [Skills/community-skills.md](Skills/community-skills.md) | Registry size and concentration, where to look (with stars), the collections worth reading, Caveman measured, security findings, how to check before installing |
 
## Where the files go
 
```text
your-project/
├── AGENTS.md                      # always loaded (Claude Code and Codex read the same file)
├── .claude/
│   ├── settings.json              # permissions, env, hooks
│   ├── agents/
│   │   └── ux-reviewer.md         # subagent
│   ├── skills/
│   │   └── stats-api/             # Skill (or a symlink to .agents/skills/stats-api)
│   │       ├── SKILL.md
│   │       ├── references/
│   │       ├── scripts/
│   │       ├── tests/
│   │       └── evals/
│   │           └── evals.json
│   └── loop.md                    # default /loop prompt
├── .codex/
│   ├── config.toml                # sandbox, approvals, [agents], [skills]
│   └── agents/
│       └── ux-reviewer.toml       # subagent
└── .agents/
    └── skills/
        └── stats-api/
            ├── SKILL.md
            └── agents/
                └── openai.yaml    # optional Codex metadata
 
~/.claude/   settings.json · agents/ · skills/ · loop.md     # user-level
~/.codex/    config.toml · agents/                           # user-level
~/.agents/   skills/                                          # user-level
```
 
## Quick lookups
 
| I want to… | Go to |
| --- | --- |
| Stop the agent reading `.env` or leaving the project | [configs](configs/README.md) |
| Decide what belongs in `AGENTS.md` vs a Skill vs a subagent vs docs | [Skills → "What you need / Where it belongs"](Skills/README.md) |
| Make a read-only reviewer agent | [creating_agents → Custom agent file](creating_agents/README.md#custom-agent-file) |
| Let one Skill serve both tools | [Skills → Claude Code vs Codex](Skills/README.md#claude-code-vs-codex) |
| Pass arguments to a Skill (`/stats-api keyword "x" 30d`) | [Skills → Invoking with arguments](Skills/README.md#invoking-with-arguments) |
| Run a Skill in its own subagent | [Skills → Running in a subagent](Skills/README.md#running-in-a-subagent) |
| Prove a Skill helps before keeping it | [evaluating-skills](Skills/evaluating-skills.md) |
| Re-run a check every N minutes / overnight | [bundled-skills → /loop](Skills/bundled-skills.md#loop) |
| Keep the agent going until tests pass | [bundled-skills → /goal](Skills/bundled-skills.md#goal) |
| Split a migration across parallel workers | [bundled-skills → /batch](Skills/bundled-skills.md#batch) |
| Vet a Skill from GitHub before installing | [community-skills → Security](Skills/community-skills.md#security) |
 
## Research referenced
 
| Paper | Finding |
| --- | --- |
| SkillsBench (arXiv 2602.12670, Feb 2026) | Curated Skills +16.2 pp average across 11 domains; self-generated Skills −1.3 pp |
| SWE-Skills-Bench (arXiv 2603.15401, Mar 2026) | 39 of 49 public SWE Skills: no change; 7 up, 3 down; avg +1.2 pp, tokens +10.5% |
| How Well Do Agentic Skills Work in the Wild? (arXiv 2604.04323, Apr 2026) | Claude loads all relevant Skills in 49% of runs, 31% with distractors |
| SpecBench (arXiv 2605.21384, May 2026) | Visible-vs-held-out test gap grows ~28 pp per 10× code size — reward hacking in long-horizon agents |
| Snyk ToxicSkills (Feb 2026) · NVIDIA SkillSpector | 76 malicious of 3,984 Skills; 13.4% critical issues · 26.1% vulnerable, 5.2% likely malicious |
