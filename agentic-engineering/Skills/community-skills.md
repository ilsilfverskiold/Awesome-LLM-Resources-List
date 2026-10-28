# Community Skills
 
Three things get conflated: a Skill being popular, developers saying it helps, and evidence it improves task outcomes.
 
## How big (Vercel, skills.sh, 25 Sep 2026)
 
| | |
| --- | --- |
| 1M | Skills in the registry by Aug 2026 |
| ~280M | Installs |
| 375 Skills (0.04%) | 62% of installs |
| Top 1.2% | 94% of installs |
| ~half of all Skills | Installed exactly once |
 
Install counts measure installs, not outcomes.
 
## Where to look
 
| Source | Stars | What | Note |
| --- | --- | --- | --- |
| [skills.sh](https://www.skills.sh/) · [vercel-labs/skills](https://github.com/vercel-labs/skills) | 37k | Registry with trending; `npx skills` installer | Claude Code and Codex |
| [anthropics/skills](https://github.com/anthropics/skills) | 184k | Anthropic's own Skills (docx, pdf, pptx, frontend-design, mcp-builder…) | Best place to see intended structure |
| [anthropics/claude-plugins-official](https://github.com/anthropics/claude-plugins-official) | 45k | Plugin marketplace — single-Skill plugins + `skill-creator` | What `/plugin install` pulls from |
| [ComposioHQ/awesome-claude-skills](https://github.com/ComposioHQ/awesome-claude-skills) | 78k | Categorized catalogue | Claude-specific |
| [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills) | 36k | 1000+ Skills across Claude Code, Codex, Cursor, Gemini CLI | Inspect each individually |
| [openai/skills](https://github.com/openai/skills) | 28k | Codex system + curated Skills | Codex packaging |
 
Many developers still find Skills via GitHub/Reddit and rewrite the idea for their own project rather than installing.
 
## Worth examining
 
| Repo | Stars | What |
| --- | --- | --- |
| [obra/superpowers](https://github.com/obra/superpowers) | 306k | Development-process Skills: brainstorming, planning, systematic-debugging, test-driven-development, executing-plans, verification-before-completion. The controversial one — praised for planning discipline, criticized for token cost and duplicating what newer models do |
| [mattpocock/skills](https://github.com/mattpocock/skills) | 304k | Engineering/code quality: grill-me, grill-with-docs, tdd, diagnosing-bugs, code-review, handoff, domain-modeling, planning |
| [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail) | 177k | Prevent overengineering: assess existing code first, inspect callers, preserve validation/error handling, don't break interfaces, leave tests for nontrivial logic |
| [JuliusBrussee/caveman](https://github.com/juliusbrussee/caveman) | 115k | Terse response style, persistent across turns. Claims 65% fewer output tokens |
| [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills) | 111k | "Production-grade engineering skills" |
| [emilkowalski/skills](https://github.com/emilkowalski/skills) | 50k | Animation: animate, animation-vocabulary, improve-animations, review-animations, apple-design — separates building an animation from evaluating one |
| [NVIDIA/skills](https://github.com/NVIDIA/skills) | 3.7k | CUDA, simulation, robotics, RAG; product-team maintained with a verification/security publication process |
 
iOS example: [ActivityKit Skill](https://github.com/dpearson2699) (Live Activities, Dynamic Island, lifecycle, layouts, push updates, review of stale/terminal states) · XcodeBuildMCP (build, run, simulator tests, screenshots, UI inspection; MCP + CLI) · OpenAI's iOS Debugger Agent Skill (uses XcodeBuildMCP).
 
### Caveman, measured
 
| | Output tokens | |
| --- | --- | --- |
| Advertised | −65% | Chat-style prose answers |
| JetBrains (82 paired coding tasks, Claude Code 2.1.200, claude-sonnet-5) | −8.5% | 592k → 542k; task success statistically indistinguishable (p = 0.82) |
 
Code, diffs, tool calls and error strings dominate agent output; Caveman leaves those unchanged.
 
## Complaints
 
- Redundant as models improve; many downloadable Skills are long and prescriptive. Developers who keep using them do so for specialized, reusable actions (a tested script that produces a spreadsheet in a fixed format), not generic coding advice.
- Stars ≠ effectiveness: SWE-Skills-Bench — 49 public SWE Skills, 39 no change, 7 up, 3 down, avg +1.2 pp; some pushed outdated or incompatible guidance.
## Security
 
| | Snyk — ToxicSkills (ClawHub + skills.sh, 5 Feb 2026) | NVIDIA — SkillSpector |
| --- | --- | --- |
| Scope | 3,984 Skills | 31,132 Skills [unverified count] |
| Malicious | 76 confirmed payloads | 5.2% likely malicious intent |
| Vulnerable | 13.4% (534) with ≥1 critical issue | 26.1%; scripts ~2× more likely |
| Checks | prompt injection, credential exposure, suspicious downloads | 64 rules / 16 categories: prompt injection, exfiltration, privilege escalation, supply chain, dangerous execution |
 
Rates are for those marketplaces and that period, not every registry.
 
### Check before installing
 
| Tool | Does | Caveat |
| --- | --- | --- |
| Snyk `mcp-scan` (`uvx mcp-scan@latest --skills`) | Scans a repo URL or Skill folder for malicious instructions and security issues | Initial check, not certification |
| [NVIDIA/SkillSpector](https://github.com/NVIDIA/SkillSpector) (22k) | Scans folders, repos, archives; structured vulnerability reports | |
| NVIDIA SkillEvaluator | Validation, semantic-overlap detection, does-it-help evaluation | Experimental |
 
Three separate stages: source review → automated scan → actual execution testing. A Skill can pass the first two and still be wrong for your project.
 
## Sources
 
- https://vercel.com/blog/state-of-agent-skills
- https://snyk.io/blog/toxicskills-malicious-ai-agent-skills-clawhub
- https://github.com/NVIDIA/SkillSpector
- https://blog.jetbrains.com/ai/2026/07/speak-to-ai-agents-like-cavemen-tosave-tokens/
- https://arxiv.org/abs/2603.15401 — SWE-Skills-Bench
- Star counts: gittrend.io, 26 Oct 2026

