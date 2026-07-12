# daystrom
Daystrom is an agent framework for easily building workflows. Give AI just enough power, but not too much. The M4 was better than the M5.

This project is still under active initial development and should not be exepected to be stable at this time.

## Quickstart

Daystrom provides convenient primitives for building AI powered workflows. The main primitives you'll want to use directly are:

| primitive  | description |
| ---------- | ----------- |
| Agent      | LLM with access to tools in a loop |
| LLM        | Base component for direct LLM interaction |
| Instructor | Structured Output component leveraging the [Instructor](https://github.com/567-labs/instructor) library |
| @tool      | decorator that wraps any function and makes it a tool |

## Tools

Daystrom comes with a few built in tools.

| Tool      | Description |
| --------- | ----------- |
| read_file | reads a file with a given offset and limit |
| web_fetch | fetches web content |
| skill     | activates a skill (see Agent Skills section) |

## Agent Skills

Daystrom agents support [Agent Skills](https://agentskills.io/home): folders containing a `SKILL.md` file with YAML frontmatter and markdown instructions.

Skills are discovered when an `Agent` is created from:

1. `./.daystrom/skills/`
2. `./.agents/skills/`
3. `~/.config/daystrom/skills/`
4. `~/.agents/skills/`

Project skills override user skills, and Daystrom-native paths override `.agents` paths within the same scope.

```text
.daystrom/skills/code-review/SKILL.md
```

```markdown
---
name: code-review
description: Reviews code changes for bugs, regressions, and missing tests. Use when asked to review code.
---

# Code Review

Focus on behavioral bugs and test gaps first.
```

By default, agents discover all available skills:

```python
agent = Agent(llm=my_llm)
```

Pass an allowlist to expose only specific skills, or an empty list to disable skills:

```python
agent = Agent(llm=my_llm, skills=["code-review"])
agent_without_skills = Agent(llm=my_llm, skills=[])
```

Agents receive a compact skills catalog in the system prompt. When a task matches a skill, the model can call the built-in `skill` tool.

Agents will use the `read_file` tool for loading local text files and skill resources. Reads are controlled by `ReadPermission`; by default, agents can read files under the current working directory and discovered skill directories only.
