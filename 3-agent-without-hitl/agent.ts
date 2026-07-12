import * as z from "zod";
import { createAgent, tool } from "langchain";

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

export const agent = createAgent({
  model: "groq:llama-3.3-70b-versatile",
  tools: [stockPriceTool, purchaseStockTool],
  systemPrompt: `You are a stock assistant. When the user asks for a stock price, always use the getStockPrice tool to fetch the live price — never guess or make up numbers. For purchase requests, use the purchaseStock tool. Keep all answers short and to the point`,
});
