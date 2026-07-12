import { HumanMessage } from "langchain";
import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

import { agent } from "./agent";

const rl = readline.createInterface({ input, output });

const threadId = "thread-1";
const config = { configurable: { thread_id: threadId } };

while (true) {
  const userInput = await rl.question("You: ");

  if (["exit", "quit"].includes(userInput.toLowerCase().trim())) {
    console.log("Goodbye!");
    rl.close();
    break;
  }

  let chatbotResponse = await agent.invoke(
    { messages: [new HumanMessage(userInput)] },
    config,
  );

  const lastMessage = chatbotResponse["messages"].at(-1);
  console.log(`\nBot: ${lastMessage?.content}\n`);
}
