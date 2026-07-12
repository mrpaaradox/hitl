# hitl

To init  project:
```bash
bun init
```

To install dependencies:
```bash
bun install
```

To install langchain:
```bash
bun add langchain @langchain/core @langchain/groq @langchain/langgraph
```

To run:
```bash
bun run index.ts
```

const png = await (agent as any).graph.getGraph().drawMermaidPng();
await Bun.write("without-hitl.png", png);# hitl
