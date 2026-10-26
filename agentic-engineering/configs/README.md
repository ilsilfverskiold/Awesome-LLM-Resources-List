# AI Coding Tools (Configs & Permissions)

Claude Code and Codex have built-in safety measures, but neither automatically knows which files in your project contain credentials that the agent should never see. • You might ask Claude to fix a login bug. It might decide to inspect configuration files, check another repository for a similar implementation, or run a script that accesses stored credentials. Those actions might be attempts to help, but they can still expose sensitive data or change things outside the intended project.

An empty settings file is quite misleading if someone assumes the agent is automatically isolated from everything sensitive.

Opening Claude Code inside a particular project in VS Code does not confine it to that project. The project folder tells Claude where to start working. It is not a security boundary. Depending on permissions, it may navigate the filesystem, read files outside the project, or run terminal commands that access other directories.

## Claude Code

### permission modes and allow / ask / deny

The permission mode determines which tools Claude can use without asking you first. Your allow, ask, and deny lists then add specific exceptions (should be set in the settings file)

| Mode | What Claude can do without asking | What happens to other actions |
| --- | --- | --- |
| default (Manual) | Read and search files, run recognized read-only commands | Asks for approval |
| acceptEdits | Read files, edit files, perform common file operations | Other commands generally require approval |
| plan | Investigate, read files, explore the codebase | Cannot edit source files until you approve the plan |
| auto | Execute actions approved by an automatic safety check | Actions can be blocked by that check |
| dontAsk | Reads and pre-approved tools | Anything that would prompt is denied |
| bypassPermissions | Everything | Nothing prompts; containers/VMs only |

Auto mode is now the built-in starting mode for interactive terminal and VS Code sessions (Claude Code v2.1.283+). default is no longer the baseline.

I think acceptEdits also auto-approves mkdir, touch, rm, rmdir, mv, cp, sed inside the working directory.

### Settings (configs/permissions)

settings.json is where you decide what Claude can do on your computer, what requires your approval, and what parts of your computer it can access. E.g. let Claude edit your app, install packages, and run tests without interrupting you, but prevent it from opening personal files, accessing AWS credentials, or sending code to external services.

| File | Use |
| --- | --- |
| `~/.claude/settings.json` | Defaults across your projects |
| `.claude/settings.json` | Shared settings for one repository |
| `.claude/settings.local.json` | Personal overrides for one repository |

To apply sandboxing and security restrictions to every Claude Code project on your computer, put them in ~/.claude/settings.json. Claude Code reads this file whenever you start a local session, including from VS Code.

| Where | Behavior |
| --- | --- |
| Main settings.json | Applies project-wide or user-wide, according to which settings file you use |
| Skill allowed-tools | Temporarily preapproves particular tools during that Skill's invocation |
| Skill disallowed-tools | Temporarily removes specified tools while the Skill is active |
| Subagent tools | Selects the tools available to that worker |

| Category | Sensible starting point |
| --- | --- |
| Reading source code | Allow within the project |
| Editing source code | Allow within the project |
| Running tests and linting | Allow normal commands |
| Git commits | Allow, when part of the workflow |
| Git push | Require approval |
| Destructive Git operations | Don't automatically approve |
| .env and local secret files | Deny reading and modification |
| SSH keys, cloud credentials, certificates | Deny access |
| Files outside the workspace | Restrict by default |
| Network access | Restrict unless needed |

#### starter settings.json

```json
{
  "model": "claude-fable-5-1[1m]",
  "effortLevel": "xhigh",
  "outputStyle": "Concise",
  "permissions": {
    "defaultMode": "acceptEdits",
    "blockReadsOutsideWorkingDirectories": true,
    "allow": [
      "WebFetch(domain:github.com)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git add *)",
      "Bash(git commit *)"
    ],
    "ask": [
      "Bash(git push *)",
      "Bash(gh pr create *)"
    ],
    "deny": [
      "Read(//**/.env)",
      "Read(//**/.env.*)",
      "Read(//**/secrets.local.json)",
      "Read(//**/secrets/**)",
      "Read(//**/*.pem)",
      "Read(//**/*.key)",
      "Read(~/.ssh/**)",
      "Read(~/.aws/**)"
    ]
  },
  "sandbox": {
    "enabled": true,
    "failIfUnavailable": true,
    "allowUnsandboxedCommands": false,
    "filesystem": {
      "denyRead": [
        "~/**/.env",
        "~/**/.env.*",
        "~/**/secrets.local.json",
        "~/**/secrets",
        "~/.ssh",
        "~/.aws"
      ],
      "denyWrite": [
        "~/**/.env",
        "~/**/.env.*",
        "~/**/secrets.local.json",
        "~/**/secrets"
      ]
    }
  }
}
```

- blockReadsOutsideWorkingDirectories was added in v2.1.257 (1 Sep 2026). It makes Read/Grep/Glob refuse outside paths and makes recognized file-reading Bash commands prompt even in auto and bypassPermissions. Open bug (Sep 2026): it ignores additionalDirectories set in project/local settings.

| Setting | Effect |
| --- | --- |
| `blockReadsOutsideWorkingDirectories` | Restricts reads outside the working project |
| `sandbox.enabled` | Runs Claude's terminal commands inside an isolated environment |
| `failIfUnavailable` | Refuses to run without the sandbox if it cannot start; prevents silently falling back |
| `allowUnsandboxedCommands` | Prevents Claude from requesting an unrestricted retry |

#### Sandboxing in Claude

On macOS, Claude Code uses Apple's built-in sandbox technology, Seatbelt. On Linux, bubblewrap. These restrict what a running program can access at the OS level. The restrictions also apply to programs launched by that command, such as Node.js or Python.

Claude's terminal sandbox is disabled by default. Even when enabled, its default read permissions cover most of the computer, including credential files such as ~/.ssh and ~/.aws/credentials. Anthropic explicitly says those credentials must be protected separately.

Sandboxing doesn't automatically mean complete project isolation:

| Access | Default sandbox behavior |
| --- | --- |
| Write to the working project | Allowed |
| Write to another project | Normally blocked |
| Read files in another project | Generally allowed |
| Read sensitive credential files | Generally allowed unless restricted |
| Connect to websites | Controlled through network restrictions |

So enabling the sandbox does not automatically stop it from reading another project or opening a .env file.

Claude's built-in file-reading tools operate outside its terminal sandbox and use their own permission rules. That's why project isolation requires configuring both the sandbox and file permissions.

Some go so far as to use a PreToolUse hook that intercepts any tool call touching .env or secrets paths and blocks it before it runs (explain hooks under tests/verifications later).

**Other stuff:**
- sandbox.credentials (v2.1.187+): dedicated block that denies reads of listed credential files (same effect as denyRead) and unsets or masks listed env vars (e.g. GITHUB_TOKEN, NPM_TOKEN) before each sandboxed command runs. Masked values are swapped back in by the sandbox proxy only on outbound requests to allowed hosts.
- Network: sandbox.network.allowedDomains lists the hosts sandboxed commands may reach; the first time a command needs a new domain, Claude Code prompts. Docs warn that allowing broad domains like github.com creates exfiltration paths.
- Windows: no native sandbox. Run Claude Code inside WSL2 with bubblewrap and socat installed.

#### test your settings

Use harmless dummy files:

1. Ask Claude to read a dummy file in the current project. Should succeed.
2. Ask it to read a dummy file in a sibling project. Should be blocked.
3. Ask it to use a terminal command to read that same sibling file. Should be blocked.
4. Ask it to read a dummy .env.local inside the current project. Should be blocked.

There are documented reports of sandbox read restrictions behaving incorrectly in particular versions, including an August 2026 report involving a subagent reading a denied .env file. So tests with harmless dummy credentials are still important.

Confirmed: issue opened 16 Aug 2026 on v2.1.233 — compound Bash command in an Agent-dispatched subagent read .env.local. Earlier related reports: May 2026, subagents don't inherit .env deny rules; May 2026 (v2.1.140), denyRead: ["//**"] not enforced.

