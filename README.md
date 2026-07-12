# HITL — Human-In-The-Loop with LangChain & LangGraph

A progressive exploration of **Human-In-The-Loop (HITL)** patterns for LLM-powered agents using [LangChain](https://js.langchain.com) and [LangGraph](https://langchain-ai.github.io/langgraphjs/). Built with [Bun](https://bun.sh) and the [Groq](https://groq.com) LLM API (`llama-3.3-70b-versatile`).

Six examples build from a simple chatbot graph through increasingly sophisticated HITL patterns where a human must **approve**, **reject**, or **edit** tool calls before they execute.

## Prerequisites

- [Bun](https://bun.sh) ≥ 1.0
- A [Groq API key](https://console.groq.com)

## Setup

```bash
bun install
echo 'GROQ_API_KEY="gsk_your_key_here"' > .env
```

## Examples

| # | Directory | Pattern | Domain | HITL |
|---|-----------|---------|--------|------|
| 1 | `1-graph-chatbot/` | Hand-rolled `StateGraph` | Stock trading | None |
| 2 | `2-graph-tool-hitl/` | `interrupt()` inside tool | Stock trading | **Yes** |
| 3 | `3-agent-without-hitl/` | `createAgent()` helper | Stock trading | None |
| 4 | `4-create-agent-hitl/` | `humanInTheLoopMiddleware` | Stock trading | **Yes** |
| 5 | `5-graph-node-hitl/` | Custom middleware graph node | Stock trading | **Yes** |
| 6 | `6-create-agent-hitl-email/` | `humanInTheLoopMiddleware` | Email sending | **Yes** |

Run any example:

```bash
bun run 1-graph-chatbot/run-graph.ts
bun run 4-create-agent-hitl/run-agent.ts
# etc.
```

## How HITL Works

1. The agent executes until it invokes a tool flagged for human approval
2. Execution **pauses** via `interrupt()` and returns an `HITLRequest` to the caller
3. The CLI runner prompts you to **approve**, **reject** (with a reason), or **edit** the arguments
4. The graph **resumes** with your decision — approved tools execute, rejected tools return an error to the LLM, edited tools run with modified arguments

## Project Structure

```
├── 1-graph-chatbot/              # Basic LangGraph chatbot (no HITL)
├── 2-graph-tool-hitl/            # HITL via interrupt() inside tool functions
├── 3-agent-without-hitl/         # Higher-level createAgent() API (no HITL)
├── 4-create-agent-hitl/          # Declarative HITL via humanInTheLoopMiddleware
├── 5-graph-node-hitl/            # Custom human-approval graph node (reference impl.)
├── 6-create-agent-hitl-email/    # Same middleware, email domain
├── .env.example                  # Environment variable template
├── package.json
└── tsconfig.json
```

## Tech Stack

| Technology | Role |
|---|---|
| [Bun](https://bun.sh) | Runtime & package manager |
| [LangChain](https://js.langchain.com) | Agent framework |
| [LangGraph](https://langchain-ai.github.io/langgraphjs/) | State-graph execution engine |
| [Groq](https://groq.com) | LLM inference (via `@langchain/groq`) |
| [Zod](https://zod.dev) | Tool input schema validation |
