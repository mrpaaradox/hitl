import * as z from "zod";

import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import {
  StateGraph,
  StateSchema,
  MessagesValue,
  START,
  END,
  type GraphNode,
  MemorySaver,
  interrupt,
  Command,
} from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import {
  initChatModel,
  SystemMessage,
  type HITLRequest,
  type HITLResponse,
} from "langchain";

// ── Tools (clean — no interrupt() inside) ──────────────────────────────────

const stockPriceTool = tool(
  async ({ symbol }) => {
    try {
      const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${encodeURIComponent(symbol)}&apikey=${process.env.ALPHAVANTAGE_API_KEY}`;
      const response = await fetch(url);
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Error fetching stock price:", error);
      return { error: "Failed to fetch stock price" };
    }
  },
  {
    name: "getStockPrice",
    description:
      "Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') using Alpha Vantage.",
    schema: z.object({
      symbol: z
        .string()
        .describe("The stock ticker symbol (e.g., AAPL, TSLA)."),
    }),
  },
);

const purchaseStockTool = tool(
  async ({ symbol, quantity }) => {
    return {
      status: "success",
      message: `Purchase order placed for ${quantity} shares of ${symbol}.`,
      symbol,
      quantity,
    };
  },
  {
    name: "purchaseStock",
    description:
      "Simulate purchasing a given quantity of a stock symbol. Requires human approval before confirming the order.",
    schema: z.object({
      symbol: z
        .string()
        .describe("The stock ticker symbol (e.g., AAPL, TSLA)."),
      quantity: z
        .number()
        .int()
        .positive()
        .describe("The number of shares to purchase."),
    }),
  },
);

const tools = [stockPriceTool, purchaseStockTool];

// ── Graph setup ────────────────────────────────────────────────────────────

const model = await initChatModel("groq:llama-3.3-70b-versatile");
const modelWithTools = model.bindTools(tools);

const MessagesState = new StateSchema({
  messages: MessagesValue,
});

const llmCall: GraphNode<typeof MessagesState> = async (state) => {
  const response = await modelWithTools.invoke([
    new SystemMessage(
      "You are a stock assistant. When the user asks for a stock price, always use the getStockPrice tool to fetch the live price — never guess or make up numbers. For purchase requests, use the purchaseStock tool. Keep all answers short and to the point.",
    ),
    ...state.messages,
  ]);
  return { messages: [response] };
};

const TOOLS_REQUIRING_APPROVAL = new Set(["purchaseStock"]);

type ToolCall = NonNullable<AIMessage["tool_calls"]>[number];

// Mirrors the internals of langchain's humanInTheLoopMiddleware:
// one interrupt() bundles every tool call needing approval, decisions map
// positionally to those calls, and the AI message's tool_calls is rebuilt so
// approved/edited calls run while rejected ones are answered with an error.
const humanApprovalNode: GraphNode<typeof MessagesState> = async (state) => {
  const lastMessage = state.messages.at(-1) as AIMessage;

  // Partition tool calls: those needing human review vs auto-approved (run as-is).
  const interruptToolCalls: ToolCall[] = [];
  const autoApprovedToolCalls: ToolCall[] = [];
  for (const toolCall of lastMessage.tool_calls ?? []) {
    if (TOOLS_REQUIRING_APPROVAL.has(toolCall.name)) {
      interruptToolCalls.push(toolCall);
    } else {
      autoApprovedToolCalls.push(toolCall);
    }
  }

  // Nothing to review → let the (auto-approved) tool calls run.
  if (interruptToolCalls.length === 0) {
    return new Command({ goto: "tools" });
  }

  // A single interrupt bundling every call that needs approval.
  const hitlRequest: HITLRequest = {
    actionRequests: interruptToolCalls.map((tc) => ({
      name: tc.name,
      args: tc.args,
      description: `Tool execution pending approval\n\nTool: ${tc.name}\nArgs: ${JSON.stringify(tc.args, null, 2)}`,
    })),
    reviewConfigs: interruptToolCalls.map((tc) => ({
      actionName: tc.name,
      allowedDecisions: ["approve", "edit", "reject"],
    })),
  };

  const { decisions } = interrupt(hitlRequest) as HITLResponse;

  // Decisions map positionally to the interrupted tool calls.
  if (!Array.isArray(decisions) || decisions.length !== interruptToolCalls.length) {
    throw new Error(
      `Number of human decisions (${decisions?.length}) does not match number of tool calls needing approval (${interruptToolCalls.length}).`,
    );
  }

  // Rebuild the AI message's tool_calls, starting from the auto-approved ones.
  const revisedToolCalls: ToolCall[] = [...autoApprovedToolCalls];
  const artificialToolMessages: ToolMessage[] = [];
  const hasRejected = decisions.some((d) => d.type === "reject");

  for (let i = 0; i < decisions.length; i++) {
    const decision = decisions[i]!;
    const toolCall = interruptToolCalls[i]!;

    if (decision.type === "approve") {
      // Keep the call unchanged — it runs with its original args.
      if (!hasRejected) revisedToolCalls.push(toolCall);
    } else if (decision.type === "edit") {
      // Replace name/args with the human's edit, reusing the original id.
      if (!hasRejected) {
        revisedToolCalls.push({
          type: "tool_call",
          name: decision.editedAction.name,
          args: decision.editedAction.args,
          id: toolCall.id,
        });
      }
    } else if (decision.type === "reject") {
      // Don't run the tool: answer the (retained) call with an error ToolMessage
      // so the model sees the rejection when we jump back to it.
      revisedToolCalls.push(toolCall);
      artificialToolMessages.push(
        new ToolMessage({
          content:
            decision.message ??
            `User rejected the tool call for \`${toolCall.name}\` with id ${toolCall.id}`,
          name: toolCall.name,
          tool_call_id: toolCall.id as string,
          status: "error",
        }),
      );
    } else {
      throw new Error(`Unsupported decision type: ${(decision as { type: string }).type}`);
    }
  }

  // Returning lastMessage with its original id replaces it via the reducer.
  lastMessage.tool_calls = revisedToolCalls;

  // Any rejection bounces the whole batch back to the model (llmCall);
  // otherwise the approved/edited calls proceed to the tools node.
  return new Command({
    update: { messages: [lastMessage, ...artificialToolMessages] },
    goto: hasRejected ? "llmCall" : "tools",
  });
};

const toolNode = new ToolNode(tools);
const checkpointer = new MemorySaver();

export const graphWithMiddlewareNode = new StateGraph(MessagesState)
  .addNode("llmCall", llmCall)
  .addNode("humanApproval", humanApprovalNode, { ends: ["tools", "llmCall"] })
  .addNode("tools", toolNode)
  .addEdge(START, "llmCall")
  .addConditionalEdges("llmCall", toolsCondition, {
    tools: "humanApproval", // intercept all tool calls through middleware first
    [END]: END,
  })
  // humanApproval routes onward itself via Command.goto (tools | llmCall).
  .addEdge("tools", "llmCall")
  .compile({ checkpointer });
