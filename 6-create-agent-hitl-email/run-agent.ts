import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

import { Command, INTERRUPT, isInterrupted } from "@langchain/langgraph";
import { HumanMessage, type HITLRequest, type HITLResponse } from "langchain";

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

  if (isInterrupted(chatbotResponse) && chatbotResponse[INTERRUPT][0]) {
    const hitlRequest = chatbotResponse[INTERRUPT][0].value as HITLRequest;
    const hitlResponse = await resolveDecision(hitlRequest);

    chatbotResponse = await agent.invoke(
      new Command({ resume: hitlResponse }),
      config,
    );
  }

  const lastMessage = chatbotResponse["messages"].at(-1);
  console.log(`\nBot: ${lastMessage?.content}\n`);
}

async function resolveDecision(
  hitlRequest: HITLRequest,
): Promise<HITLResponse> {
  const actionRequest = hitlRequest.actionRequests[0];
  const allowed = hitlRequest.reviewConfigs[0]?.allowedDecisions ?? [];

  console.log(`\nHuman Approval Needed: ${actionRequest?.description}`);
  console.log(`Options:\n${allowed.join("\n")}`);

  while (true) {
    const choice = (await rl.question("CHOOSE: ")).trim().toLowerCase();

    if (!(allowed as string[]).includes(choice)) {
      console.log(
        `Invalid input "${choice}". Please choose one of: ${allowed.join(", ")}`,
      );
      continue;
    }

    if (choice === "approve") {
      return { decisions: [{ type: "approve" }] };
    }

    if (choice === "reject") {
      const message = (await rl.question("Reject Message: ")).trim();
      return { decisions: [{ type: "reject", message }] };
    }

    // choice === "edit"
    const recipient = (await rl.question("Enter New Recipient Address: ")).trim();

    return {
      decisions: [
        {
          type: "edit",
          editedAction: {
            name: actionRequest?.name as string,
            args: {
              ...actionRequest?.args,
              recipient,
            },
          },
        },
      ],
    };
  }
}
