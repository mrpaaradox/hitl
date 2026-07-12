import {
  initChatModel,
  SystemMessage,
  type HITLRequest,
  type HITLResponse,
} from "langchain";

import {
  StateGraph,
  StateSchema,
  MessagesValue,
  START,
  END,
  type GraphNode,
  MemorySaver,
} from "@langchain/langgraph";

import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";

import * as z from "zod";
import { tool } from "@langchain/core/tools";
import { interrupt } from "@langchain/langgraph";

export const stockPriceTool = tool(
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

export const purchaseStockTool = tool(
  async ({ symbol, quantity }) => {
    const hitlRequest: HITLRequest = {
      actionRequests: [
        {
          name: "purchaseStock",
          args: {
            symbol,
            quantity,
          },
          description: `Tool execution pending approval\n\nTool: purchaseStock\nArgs: {\n  \"symbol\": \"${symbol}\",\n  \"quantity\": ${quantity}\n}"`,
        },
      ],
      reviewConfigs: [
        {
          actionName: "purchaseStock",
          allowedDecisions: ["approve", "edit", "reject"],
        },
      ],
    };

    const hitlResponse: HITLResponse = interrupt(hitlRequest);
    const responseType = hitlResponse.decisions[0]?.type;

    if (responseType === "approve") {
      return {
        status: "success",
        message: `Purchase order placed for ${quantity} shares of ${symbol}.`,
        symbol,
        quantity,
      };
    } else if (responseType === "reject") {
      return {
        status: "cancelled",
        message: hitlResponse.decisions[0]?.message,
        symbol,
        quantity,
      };
    } else if (responseType === "edit") {
      return {
        status: "success",
        message: `Purchase order placed for ${hitlResponse.decisions[0]?.editedAction.args["quantity"]} shares of ${hitlResponse.decisions[0]?.editedAction.args["symbol"]}.`,
        symbol: hitlResponse.decisions[0]?.editedAction.args["symbol"],
        quantity: hitlResponse.decisions[0]?.editedAction.args["quantity"],
      };
    } else {
      return {
        status: "cancelled",
        message: `Received invalid arguments`,
        symbol,
        quantity,
      };
    }
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

export const toolsByName = {
  [stockPriceTool.name]: stockPriceTool,
  [purchaseStockTool.name]: purchaseStockTool,
};

export type ToolName = keyof typeof toolsByName;
export const tools = Object.values(toolsByName);

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
  return {
    messages: [response],
  };
};

const toolNode = new ToolNode(tools);

const checkpointer = new MemorySaver();

export const graph = new StateGraph(MessagesState)
  .addNode("llmCall", llmCall)
  .addNode("tools", toolNode)
  .addEdge(START, "llmCall")
  .addConditionalEdges("llmCall", toolsCondition, {
    tools: "tools",
    [END]: END,
  })
  .addEdge("tools", "llmCall")
  .compile({ checkpointer: checkpointer });
