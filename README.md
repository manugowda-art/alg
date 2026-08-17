# ALG

**A**gent Harness -> **L**oop Engineering -> **G**raph Engineering. 

environment → feedback → flow.

# What do they do
1. Harness engineering builds the machinery around the model.
2. Loop engineering designs the repeated work-and-feedback cycle.
3. Graph engineering makes the workflow topology explicit: nodes, branches, joins, state transitions and controlled cycles.


# Agent Architecture Comparison

| Question | Agent harness | Loop engineering | Graph engineering |
| :--- | :--- | :--- | :--- |
| **Primary concern** | Operational capability | Iterative progress and feedback | Explicit control flow |
| **Core object** | Model wrapper / runtime | A bounded repeatable cycle | A directed graph of steps |
| **Typical building blocks** | Tools, memory, sandbox, middleware, permissions, traces | Trigger, goal, action, evidence, feedback, stop rule | Nodes, edges, shared state, branches, joins, interrupts, cycles |
| **Failure it fixes** | "The model cannot safely do the work." | "The agent stops too early or repeats weak work." | "The workflow is hard to reason about or control." |
| **Best fit** | General agent platform or task-specific runtime | Open-ended work that improves through verification | Complex multi-step processes with known decision points |
| **Main risk** | A bloated, opaque runtime | Infinite retries, token burn, reward hacking | Over-engineered diagrams and brittle paths |
