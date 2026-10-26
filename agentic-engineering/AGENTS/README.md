# Building an AGENTS.md file & How to set up Rules

## The main AGENTS.md file
 
The ideal `AGENTS.md` or `CLAUDE.md` is not a comprehensive guide to the project. It's a small set of persistent instructions that helps the agent work correctly without unnecessarily restricting how it solves problems (these will go into a project's source of truth (docs) later).

In 2026, you only need one `AGENTS.md` file and Claude and Codex will read from the same one. 

```markdown
project/
│
├── AGENTS.md # always loaded
├── CLAUDE.md # now optional
```
### What developers actually put in these files
 
There's useful new evidence from Stride Research's September 2026 analysis of 7,370 public GitHub repositories (verified: [census published 27 September 2026](https://www.stride.page/research/agents-md-claude-md-study-2026); AGENTS.md at the root in 25.3%, CLAUDE.md in 19.2%).
 
| Finding | Share |
|---|---|
| Had at least one agent instruction file | 33.4% |
| Of `AGENTS.md` files named a test, build, or lint command | 73.5% |
| Mentioned security or secrets | 39.3% |
| Explicitly prohibited leaking secrets | 10.6% |
| Median `AGENTS.md` length | 819 words (verified: 105 lines; Codex's `/init` asks for 200–400 words, and Codex reads at most 32 KiB of instructions by default) |
 
The study inspected active public repositories with at least 5,000 GitHub stars, so it reflects large open-source projects rather than all software development.
 
The five most common categories were:
 
1. Testing instructions (verified: 55.7%)
2. Code style and conventions (54.3%)
3. Project structure (54.3%)
4. Git and pull-request procedures (47.6%)
5. Commands (44.0%) [corrected: was "Build and development commands" – the study counts build separately at 29.1%]
What people commonly write is therefore not necessarily what research suggests they should write.
 
### What's worth including
 
I would organize the content like this:
 
| Content | Worth including? | Example |
|---|---|---|
| Project identity | Briefly | "This is a Next.js application with a separate Node.js API." |
| Commands | Yes, especially nonstandard ones | `pnpm test:api` |
| Architecture invariants | Yes, if broadly relevant | "The server owns subscription state." |
| Existing utilities | When agents repeatedly duplicate them | "Reuse the HTTP client in `src/lib/api.ts`." |
| Verification | Yes, make it executable | "Run `pnpm typecheck` after TS changes." |
| Git workflow | If the team has a specific convention | "Commit completed changes; never push without approval." |
| Documentation references | Yes, selectively | "For database changes, consult `docs/database.md`." |
| Scope control | Sometimes | "Don't fix unrelated issues discovered during a task." |
 
- Meaningless to put in these files: describing how to write software generally.
- Current task progress, lengthy architectural histories, deployment playbooks, full API documentation, giant repo maps, and accumulated one-off bug fixes don't belong in the always-loaded root instructions. We'll cover where you should put these in a bit.

### Examples from public repos
 
**Vercel Labs: agent-browser/AGENTS.md** [unverified]
 
Vercel's file specifies pnpm, CLI naming conventions, where command-line color handling lives, and a concrete requirement to keep CLI and MCP functionality synchronized.
 
The strongest example is a repository-specific constraint: when CLI functionality changes, the corresponding MCP implementation must be updated too. That's the kind of requirement an agent might otherwise overlook.
 
**pnpm: AGENTS.md** [unverified]
 
pnpm documents project-specific conventions and links to the canonical development setup. It also has guidance about comments and documentation: avoid repeating what the code already explains, and put shared reasoning in one place.


