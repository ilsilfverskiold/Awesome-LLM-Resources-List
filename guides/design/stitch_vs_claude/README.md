# Google Stitch vs Claude Code Skill for Frontend Design

A one-shot comparison of two systems generating a landing page for the same product brief.

## Product brief used (Zero-shot)

**Name:** Looma  
**What it is:** An AI operating layer for internal company knowledge, workflows, and agents. It connects to docs, tickets, Slack, CRMs, and databases, then lets teams search, act, and automate across them.  
**Audience:** Mid-size to enterprise companies. Buyers are CTOs, Heads of Ops, and technical product leaders. Users are internal teams.  
**Core tension:** The product is technically advanced, but the landing page must not feel cold, sterile, or overloaded with jargon.

## Direct comparison (Stitch to the left, Claude Code with the design skill to the right).

![hero comparison](./mobile_hero.png)

## Mobile and Desktop 

### Stitch

![Desktop](./stitch_desktop_full.png)
![Mobile](./stitch_mobile_full.png)

### Claude Code (w/ Frontend-design Skill)

![Desktop](./claude_skill_desktop_full.png)
![Mobile](./claude_skill_mobile_full.png)

## Files in this repo

- `mobile_hero.png` - hero-only comparison, Stitch on the left and Claude skill on the right
- `stitch_desktop_full.png`
- `claude_skill_desktop_full.png`
- `stitch_mobile_full.png`
- `claude_skill_mobile_full.png`

## What the two tools are

### Google Stitch

Google Stitch is a design-oriented product from Google Labs. Google describes it as a tool that turns natural language into high-fidelity UI and, more recently, as an AI-native design canvas for creating, iterating, and collaborating on interface design.

In plain terms: Stitch is meant to generate UI concepts directly. Design is the product.

### Claude Code + a frontend design skill

Claude Code is an agentic coding tool. Anthropic describes it as a tool that can read a codebase, edit files, run commands, and work across development tools. A `SKILL.md` file adds reusable instructions that extend how Claude approaches a task.

In plain terms: Claude Code is not a dedicated design product. It is a coding agent that can be pushed into design behavior through prompt structure, process rules, references, and output constraints.

## Why both likely produce similar designs from the same prompt

LLMs and model-based design tools do not invent from a vacuum. They predict likely next outputs based on patterns they have learned. When the category is narrow, the prompt is strong, and the task is one-shot, outputs tend to cluster.

### Both systems have learned the same broad web priors

Even if the underlying models and product experiences differ, both systems are drawing on a huge body of existing interface patterns.

That means they have both likely absorbed the same common priors around:

- hero-first SaaS layout
- card sections
- integrations rows
- testimonial blocks
- security sections
- visual metaphors for intelligence, networks, or flow

So the same prompt will often trigger the same general answer class.

### One-shot generation increases convergence

When you do one-shot design, the model usually picks the most statistically plausible good answer.

It does not get a long critique loop where you can say:

- make it less expected
- strip out the cliché AI glow
- make it feel more like enterprise editorial software
- push away from standard SaaS composition

Without that iterative pressure, many systems fall back to the safest high-probability layout.

### Skills steer behavior, but they do not replace model priors

A frontend design skill can improve process and shape taste. It can push Claude toward certain structures, references, and output standards.

But it still works on top of the model's learned priors. So it can bend the result, not fully escape the space of likely landing-page answers.

## References

Official sources used for tool descriptions:

- Google Labs / Google Blog on Stitch
  - https://blog.google/innovation-and-ai/models-and-research/google-labs/stitch-ai-ui-design/
  - https://blog.google/innovation-and-ai/products/google-io-2025-all-our-announcements/
  - https://labs.google/experiments?category=develop

- Anthropic docs on Claude Code and skills
  - https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview
  - https://docs.anthropic.com/en/docs/claude-code/slash-commands
  - https://console.anthropic.com/docs/en/build-with-claude/skills-guide
