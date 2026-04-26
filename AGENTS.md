# AGENTS.md

## Purpose of this file

This file defines how an Agent should work in this project.
It provides persistent context about working style, boundaries, testing expectations, and how to report progress at the end of a task.

The specific task will always be provided in the prompt.
This file describes the general working approach for the project.

---

## General working principles

* Make small, focused, and clear changes.
* Prefer simple and readable solutions over overly clever ones.
* Before making a significant change, read the existing code and understand the patterns already used in the project.
* Stay consistent with the existing code style, structure, and naming conventions.
* Do not perform broad refactors unless they are directly required for the task.
* Do not change existing behavior without a clear reason and without mentioning it.
* If there are several reasonable ways to solve a problem, prefer the least invasive one.
* When there is meaningful uncertainty, stop and ask before making an architectural or high-risk change.
* Search for references online when relevant.

---

## Approach to solving tasks

When receiving a task:

1. Read the prompt carefully and understand the goal.

2. Identify the relevant files, modules, and tests.

3. Study the existing implementation before changing it.

4. Form a short plan before making a significant change.

5. Make the smallest change that satisfies the requirement.

---

## Code style

* Write code that is readable, direct, and easy to maintain.

* Use clear names that describe the business or technical intent.

* Avoid unclear abbreviations.

* Avoid adding a new abstraction unless there is a real need for it.

* Prefer small functions with clear responsibility.

* Do not add a new dependency unless it is truly necessary. If a dependency is added, explain why.

* Do not introduce large formatting changes in files that are unrelated to the task.

---

## Preserving existing behavior

* Preserve backward compatibility unless the prompt explicitly asks otherwise.
* Do not change an existing public API, contract, schema, configuration, or behavior without a clear need.
* If a breaking change is required, state it explicitly and explain the impact.
* Do not delete code, tests, or logging without understanding why they exist.
* Be especially careful with error handling, observability, metrics, retries, and fallbacks.

---

## Testing and validation

* Do not add tests on your own.

---

## Communication and final summary

At the end of each task, report concisely:

* What changed.
* Which files were modified.
* Which tests or manual checks were performed.
* Whether there are any risks, assumptions, or incomplete items.

The summary should be clear, practical, and short.
There is no need to explain every small change if it is clear from the diff.

---

## Personal working preferences

* I prefer practical and simple solutions over unnecessarily complex ones.
* I prefer small, focused diffs that are easy to understand and review.
* It is important for me to understand tradeoffs when there are several good options.
* I prefer the Agent to state uncertainty instead of guessing confidently.
* I prefer the Agent not to make broad or architectural changes without first explaining the direction.
* I prefer the Agent to respect the existing project structure before proposing a new one.

---

## Do not do the following without explicit instruction

* Do not commit or push.
* Do not change a public API.
* Do not replace libraries or frameworks.
* Do not perform broad refactors.
* Do not change production config or deployment files.
* Do not remove existing tests just to make the test suite pass.
* Do not add a new dependency without justification.
* Do not perform broad formatting changes that are unrelated to the task.

---

## When the prompt is unclear

If the task is unclear:

* Ask for clarification.
* Try to infer the most reasonable intent from the code and context.
* If it is a small and safe decision, proceed and mention the assumption in the summary.
* If it is a decision with significant impact, ask a focused question before making the change.

---

## Relevant repositories for vLLM-related work

When working on tasks related to vLLM, use the repositories below as references when relevant.
The local paths should be filled in by the developer.

```text
vLLM repository:
/swgwork/rtourgeman/vllm_deep_ep_nixl/vllm

NIXL repository:
/swgwork/rtourgeman/nixl

NIXL-EP:
/swgwork/rtourgeman/nixl/examples/device/ep

DeepEP:
/swgwork/rtourgeman/DeepEP
```
