import * as z from "zod";
import { MemorySaver } from "@langchain/langgraph";
import { createAgent, humanInTheLoopMiddleware, tool } from "langchain";

const USERS = {
  sujoy: {
    name: "Sujoy",
    email: "sujoy@codersgyan.com",
    type: "customer",
  },
  rakesh: {
    name: "Rakesh",
    email: "rakesh@codersgyan.com",
    type: "customer",
  },
  ajay: {
    name: "Ajay",
    email: "ajay@codersgyan.com",
    type: "premium_customer",
  },
};

// Create email tools
const getUserEmail = tool(
  async (input) => {
    return USERS[input.username as keyof typeof USERS];
  },
  {
    name: "get_user_email",
    description: "Get the email address of a user",
    schema: z.object({
      username: z.string(),
    }),
  },
);

const sendEmailTool = tool(
  async (input) => {
    // In a real implementation, this would send the email
    // For now, we just return a success message
    console.log(`Email sent successfully to ${input.recipient}`);

    return {
      success: true,
      message: `Email sent successfully to ${input.recipient}`,
      email: input,
    };
  },
  {
    name: "send_email",
    description:
      "Send an email to someone. This requires human approval before executing.",
    schema: z.object({
      recipient: z.string(),
      subject: z.string(),
      body: z.string(),
    }),
  },
);

export const agent = createAgent({
  model: "groq:llama-3.3-70b-versatile",
  tools: [getUserEmail, sendEmailTool],
  middleware: [
    humanInTheLoopMiddleware({
      interruptOn: {
        send_email: true,
      },
    }),
  ],
  checkpointer: new MemorySaver(),
  systemPrompt: `You are an email sending assistant. Your job is to help the user compose and send emails to people.

When the user wants to email someone, first use the get_user_email tool to look up their email address by username — never guess or make up an email address. Use the details it returns (name, email) to address the message correctly.

To send an email, use the send_email tool with a clear recipient, subject, and body. Sending an email requires human approval before it goes out, so write a complete, well-formed message the user can review and approve.

Keep your own replies short and to the point.`,
});

