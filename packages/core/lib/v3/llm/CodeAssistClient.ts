/**
 * AUDITARIA: OAuth-compatible LLM Client using Google Code Assist API
 *
 * This client uses authClient.request() directly to make API calls to the
 * Code Assist API (cloudcode-pa.googleapis.com), following the same pattern
 * as Auditaria's CodeAssistServer. This is the correct endpoint for OAuth
 * authentication - Vertex AI requires the API to be enabled on the project,
 * while Code Assist API works with OAuth out of the box.
 */

import type { AuthClient } from "google-auth-library";
import { Schema } from "@google/genai";
import { LogLine } from "../types/public/logs";
import { AvailableModel } from "../types/public/model";
import { toGeminiSchema } from "../../utils";
import {
  ChatCompletionOptions,
  ChatMessage,
  CreateChatCompletionOptions,
  LLMClient,
  LLMResponse,
  AnnotatedScreenshotText,
} from "./LLMClient";
import { StagehandError } from "../types/public/sdkErrors";

// Code Assist API constants (same as Auditaria's CodeAssistServer)
const CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com";
const CODE_ASSIST_API_VERSION = "v1internal";

// Code Assist API types
interface CodeAssistPart {
  text?: string;
  inlineData?: {
    mimeType: string;
    data: string;
  };
  functionCall?: {
    name: string;
    args: Record<string, unknown>;
  };
}

interface CodeAssistContent {
  role: string;
  parts: CodeAssistPart[];
}

interface CodeAssistGenerationConfig {
  maxOutputTokens?: number;
  temperature?: number;
  topP?: number;
  responseMimeType?: string;
  responseSchema?: Schema; // Schema for structured output (Gemini format)
}

// Inner request format (nested inside "request" field)
interface CodeAssistInnerRequest {
  contents: CodeAssistContent[];
  generationConfig?: CodeAssistGenerationConfig;
  tools?: Array<{
    functionDeclarations: Array<{
      name: string;
      description: string;
      parameters: unknown;
    }>;
  }>;
}

// Outer request format (what we send to the API)
interface CodeAssistRequest {
  model: string;
  project?: string;
  user_prompt_id?: string;
  request: CodeAssistInnerRequest;
}

// Response types
interface CodeAssistCandidate {
  content?: {
    role: string;
    parts: CodeAssistPart[];
  };
  finishReason?: string;
}

interface CodeAssistUsageMetadata {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  totalTokenCount?: number;
}

interface CodeAssistInnerResponse {
  candidates?: CodeAssistCandidate[];
  usageMetadata?: CodeAssistUsageMetadata;
}

// Outer response format (what we receive from the API)
interface CodeAssistResponse {
  response: CodeAssistInnerResponse;
  traceId?: string;
}

// Role mapping from Stagehand to Code Assist API
const roleMap: { [key in ChatMessage["role"]]: string } = {
  user: "user",
  assistant: "model",
  system: "user", // System messages prepended to first user message
};

export class CodeAssistClient extends LLMClient {
  public type = "google-oauth" as const;
  public hasVision: boolean = true;
  private authClient: AuthClient;
  private project: string;
  private logger: (message: LogLine) => void;

  constructor({
    logger,
    modelName,
    authClient,
    project,
  }: {
    logger: (message: LogLine) => void;
    modelName: AvailableModel;
    authClient: AuthClient;
    project: string;
    location?: string; // Kept for compatibility but not used by Code Assist API
  }) {
    super(modelName);
    this.authClient = authClient;
    this.project = project;
    this.logger = logger;
    this.modelName = modelName;
    this.clientOptions = {};

    this.logger({
      category: "code-assist-client",
      message: `Initialized CodeAssistClient for OAuth mode (Code Assist API)`,
      level: 1,
      auxiliary: {
        project: { value: this.project, type: "string" },
        model: { value: modelName, type: "string" },
        endpoint: { value: CODE_ASSIST_ENDPOINT, type: "string" },
      },
    });
  }

  /**
   * Build the Code Assist API endpoint URL
   */
  private getEndpointUrl(): string {
    // Code Assist API format: https://cloudcode-pa.googleapis.com/v1internal:generateContent
    return `${CODE_ASSIST_ENDPOINT}/${CODE_ASSIST_API_VERSION}:generateContent`;
  }

  /**
   * Convert Stagehand ChatMessage[] to Code Assist Content[]
   */
  private formatMessages(
    messages: ChatMessage[],
    image?: ChatCompletionOptions["image"]
  ): CodeAssistContent[] {
    // AUDITARIA DEBUG: Log incoming messages
    // console.log('[CodeAssistClient] formatMessages() input:', JSON.stringify(messages, null, 2));

    const contents: CodeAssistContent[] = [];
    let systemInstruction: string | null = null;

    messages.forEach((msg, index) => {
      const role = roleMap[msg.role];
      if (!role) {
        this.logger({
          category: "code-assist-client",
          message: `WARNING: Unsupported role: ${msg.role}`,
          level: 1,
        });
        return;
      }

      // Collect system messages to prepend to first user message
      if (msg.role === "system") {
        if (typeof msg.content === "string") {
          systemInstruction =
            (systemInstruction ? systemInstruction + "\n\n" : "") + msg.content;
        }
        return;
      }

      const parts: CodeAssistPart[] = [];

      if (Array.isArray(msg.content)) {
        msg.content.forEach((partContent) => {
          if (partContent.type === "text") {
            parts.push({ text: partContent.text });
          } else if (partContent.type === "image_url") {
            if ("image_url" in partContent && partContent.image_url?.url) {
              const base64Data = partContent.image_url.url.split(",")[1];
              const mimeTypeMatch = partContent.image_url.url.match(
                /^data:(image\/\w+);base64,/
              );
              if (base64Data && mimeTypeMatch) {
                parts.push({
                  inlineData: { mimeType: mimeTypeMatch[1], data: base64Data },
                });
              }
            }
          }
        });
      } else if (typeof msg.content === "string") {
        // AUDITARIA: Don't add empty text parts - they cause issues with Code Assist API
        if (msg.content.trim()) {
          parts.push({ text: msg.content });
        }
      }

      // AUDITARIA: Handle tool_calls for assistant messages
      // Convert to Code Assist API's functionCall format
      if (msg.role === "assistant" && (msg as any).tool_calls) {
        const toolCalls = (msg as any).tool_calls;
        for (const toolCall of toolCalls) {
          parts.push({
            functionCall: {
              name: toolCall.function.name,
              args: JSON.parse(toolCall.function.arguments),
            },
          });
        }
      }

      // Add image from options if this is the last message and it's a user message
      if (image && index === messages.length - 1 && msg.role === "user") {
        const imageDesc = image.description || AnnotatedScreenshotText;
        parts.push({ text: imageDesc });
        parts.push({
          inlineData: {
            mimeType: "image/jpeg",
            data: image.buffer.toString("base64"),
          },
        });
      }

      // Apply system instruction to the first non-system message
      if (systemInstruction && contents.length === 0 && role === "user") {
        const firstPartText = parts.find((p) => "text" in p && p.text);
        if (firstPartText && "text" in firstPartText) {
          firstPartText.text = `${systemInstruction}\n\n${firstPartText.text}`;
        } else {
          parts.unshift({ text: systemInstruction });
        }
        systemInstruction = null;
      }

      if (parts.length > 0) {
        contents.push({ role, parts });
      }
    });

    // If system instruction wasn't applied, add it as first user message
    if (systemInstruction) {
      contents.unshift({ role: "user", parts: [{ text: systemInstruction }] });
    }

    // AUDITARIA DEBUG: Log formatted output
    // console.log('[CodeAssistClient] formatMessages() output:', JSON.stringify(contents, null, 2));

    return contents;
  }

  async createChatCompletion<T = LLMResponse>({
    options,
    logger,
    retries = 3,
  }: CreateChatCompletionOptions): Promise<T> {
    const {
      image,
      requestId,
      response_model,
      temperature,
      top_p,
      maxOutputTokens,
      tools,
      tool_choice,
    } = options;

    const formattedMessages = this.formatMessages(options.messages, image);

    const generationConfig: CodeAssistGenerationConfig = {
      maxOutputTokens: maxOutputTokens,
      temperature: temperature,
      topP: top_p,
    };

    // If response_model is provided, request JSON output with schema
    if (response_model) {
      generationConfig.responseMimeType = "application/json";
      const geminiSchema = toGeminiSchema(response_model.schema);
      generationConfig.responseSchema = geminiSchema;
    }

    // Build the Code Assist API request format (wrapped structure)
    const requestPayload: CodeAssistRequest = {
      model: this.modelName,
      project: this.project,
      user_prompt_id: requestId || `stagehand-${Date.now()}`,
      request: {
        contents: formattedMessages,
        generationConfig,
        tools: tools
          ? [
              {
                functionDeclarations: tools.map((t) => ({
                  name: t.function.name,
                  description: t.function.description,
                  parameters: t.function.parameters,
                })),
              },
            ]
          : undefined,
      },
    };

    const endpointUrl = this.getEndpointUrl();

    // AUDITARIA DEBUG: Log request payload before sending
    // console.log('[CodeAssistClient] Request payload:', JSON.stringify(requestPayload, null, 2));

    logger({
      category: "code-assist-client",
      message: "Creating chat completion via Code Assist API",
      level: 2,
      auxiliary: {
        modelName: { value: this.modelName, type: "string" },
        requestId: { value: requestId || "unknown", type: "string" },
        endpoint: { value: endpointUrl, type: "string" },
      },
    });

    try {
      // Make request using authClient.request() - this automatically handles OAuth tokens
      const response = await this.authClient.request({
        url: endpointUrl,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestPayload),
      });

      // Parse the wrapped response format
      const result = response.data as CodeAssistResponse;
      const innerResponse = result.response;

      logger({
        category: "code-assist-client",
        message: "Received response from Code Assist API",
        level: 2,
        auxiliary: {
          requestId: { value: requestId || "unknown", type: "string" },
          traceId: { value: result.traceId || "unknown", type: "string" },
          candidateCount: {
            value: String(innerResponse?.candidates?.length || 0),
            type: "string",
          },
        },
      });

      // Extract content and function calls from response
      const candidate = innerResponse?.candidates?.[0];
      const finishReason = candidate?.finishReason || "unknown";
      let content: string | null = null;
      const toolCalls: Array<{
        id: string;
        type: string;
        function: { name: string; arguments: string };
      }> = [];

      if (candidate?.content?.parts && Array.isArray(candidate.content.parts)) {
        // Extract text parts
        const textParts = candidate.content.parts
          .filter((p) => p.text)
          .map((p) => p.text);
        content = textParts.join("");

        // Extract function calls
        candidate.content.parts
          .filter((p) => p.functionCall)
          .forEach((p, index) => {
            if (p.functionCall) {
              toolCalls.push({
                id: `call_${index}_${Date.now()}`,
                type: "function",
                function: {
                  name: p.functionCall.name,
                  arguments: JSON.stringify(p.functionCall.args),
                },
              });
            }
          });
      }

      // Construct LLMResponse shape (same as GoogleClient)
      const llmResponse: LLMResponse = {
        id: result.traceId || requestId || `code-assist-${Date.now()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: this.modelName,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: content,
              tool_calls: toolCalls,
            },
            finish_reason: finishReason,
          },
        ],
        usage: {
          prompt_tokens: innerResponse?.usageMetadata?.promptTokenCount || 0,
          completion_tokens: innerResponse?.usageMetadata?.candidatesTokenCount || 0,
          total_tokens: innerResponse?.usageMetadata?.totalTokenCount || 0,
        },
      };

      // Handle response_model validation (structured output)
      if (response_model) {
        let parsedData;
        try {
          const potentialJson =
            content?.trim().replace(/^```json\n?|\n?```$/g, "") || "{}";
          parsedData = JSON.parse(potentialJson);
        } catch (e) {
          logger({
            category: "code-assist-client",
            message: `Failed to parse JSON response: ${(e as Error).message}`,
            level: 0,
            auxiliary: {
              content: { value: content || "null", type: "string" },
            },
          });
          if (retries > 0) {
            return this.createChatCompletion({
              options,
              logger,
              retries: retries - 1,
            });
          }
          throw new StagehandError(
            `Failed to parse JSON response: ${(e as Error).message}`
          );
        }

        // Return extraction result format
        const extractionResult = {
          data: parsedData,
          usage: llmResponse.usage,
        };

        return extractionResult as T;
      }

      return llmResponse as T;
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);

      logger({
        category: "code-assist-client",
        message: `Error during Code Assist API call: ${errorMessage}`,
        level: 0,
        auxiliary: {
          requestId: { value: requestId || "unknown", type: "string" },
          error: { value: errorMessage, type: "string" },
        },
      });

      // Retry logic
      if (retries > 0) {
        logger({
          category: "code-assist-client",
          message: `Retrying... (${retries} attempts left)`,
          level: 1,
        });
        await new Promise((resolve) =>
          setTimeout(resolve, 1000 * (4 - retries))
        );
        return this.createChatCompletion({
          options,
          logger,
          retries: retries - 1,
        });
      }

      if (error instanceof StagehandError) {
        throw error;
      }
      throw new StagehandError(`Code Assist API request failed: ${errorMessage}`);
    }
  }
}
