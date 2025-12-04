import { createAgentTools } from "../agent/tools";
import { LogLine } from "../types/public/logs";
import { V3 } from "../v3";
import {
  ModelMessage,
  ToolSet,
  wrapLanguageModel,
  stepCountIs,
  type LanguageModelUsage,
  type StepResult,
} from "ai";
import { processMessages } from "../agent/utils/messageProcessing";
import { LLMClient, ChatMessage, LLMResponse } from "../llm/LLMClient"; // AUDITARIA
import { LLMTool } from "../types/public/model"; // AUDITARIA
import {
  AgentExecuteOptions,
  AgentResult,
  AgentContext,
  AgentState,
  AgentStreamResult,
  AgentStepUpdate, // AUDITARIA
} from "../types/public/agent";
import { V3FunctionName } from "../types/public/methods";
import { mapToolResultToActions } from "../agent/utils/actionMapping";
import { MissingLLMConfigurationError } from "../types/public/sdkErrors";
import { zodToJsonSchema } from "zod-to-json-schema"; // AUDITARIA

export class V3AgentHandler {
  private v3: V3;
  private logger: (message: LogLine) => void;
  private llmClient: LLMClient;
  private executionModel?: string;
  private systemInstructions?: string;
  private mcpTools?: ToolSet;

  constructor(
    v3: V3,
    logger: (message: LogLine) => void,
    llmClient: LLMClient,
    executionModel?: string,
    systemInstructions?: string,
    mcpTools?: ToolSet,
  ) {
    this.v3 = v3;
    this.logger = logger;
    this.llmClient = llmClient;
    this.executionModel = executionModel;
    this.systemInstructions = systemInstructions;
    this.mcpTools = mcpTools;
  }

  private isOAuthMode(): boolean {
    return this.llmClient.type === "google-oauth";
  } // AUDITARIA": Check if we're in OAuth mode,  OAuth mode requires direct API calls (createChatCompletion) instead of AI SDK

  private async prepareAgent(
    instructionOrOptions: string | AgentExecuteOptions,
  ): Promise<AgentContext> {
    try {
      const options =
        typeof instructionOrOptions === "string"
          ? { instruction: instructionOrOptions }
          : instructionOrOptions;

      const maxSteps = options.maxSteps || 20;

      const systemPrompt = this.buildSystemPrompt(
        options.instruction,
        this.systemInstructions,
      );
      const tools = this.createTools();
      const allTools: ToolSet = { ...tools, ...this.mcpTools };
      const messages: ModelMessage[] = [
        { role: "user", content: options.instruction },
      ];

      // AUDITARIA_MODIFY_START : Skip getLanguageModel check for OAuth mode
      // OAuth mode uses direct API calls via createChatCompletion()
      let wrappedModel;
      if (!this.isOAuthMode()) { // AUDITARIA_MODIFY_END
        if (!this.llmClient?.getLanguageModel) {
          throw new MissingLLMConfigurationError();
        }
        const baseModel = this.llmClient.getLanguageModel();
        wrappedModel = wrapLanguageModel({ // AUDITARIA_MODIFY
          model: baseModel,
          middleware: {
            transformParams: async ({ params }) => {
              const { processedPrompt } = processMessages(params);
              return { ...params, prompt: processedPrompt } as typeof params;
            },
          },
        });
      } //AUDITARIA_MODIFY

      const initialPageUrl = (await this.v3.context.awaitActivePage()).url();

      return {
        options,
        maxSteps,
        systemPrompt,
        allTools,
        messages,
        wrappedModel,
        initialPageUrl,
      };
    } catch (error) {
      this.logger({
        category: "agent",
        message: `failed to prepare agent: ${error}`,
        level: 0,
      });
      throw error;
    }
  }

  // AUDITARIA_MODIFY_START : Enhanced step handler for progress updates and pause support
  private createStepHandler(
    state: AgentState,
    onStep?: (step: AgentStepUpdate) => void,
    checkPauseState?: () => Promise<void>,
  ) {
    let stepNumber = 0;
    return async (event: StepResult<ToolSet>) => {
      stepNumber++;

      // AUDITARIA: Emit "starting" update BEFORE processing tool calls
      // This allows UI to show which step is currently executing
      if (onStep) {
        onStep({
          stepNumber,
          actions: [],
          message: "", // No reasoning yet - step just started
          completed: false, // Signals step is starting/executing
          currentUrl: state.currentPageUrl,
        });
      }
      // AUDITARIA_MODIFY_END

      this.logger({
        category: "agent",
        message: `Step finished: ${event.finishReason}`,
        level: 2,
      });

      // AUDITARIA_FEATURE_START: Check pause state at step boundary
      if (checkPauseState) {
        try {
          await checkPauseState();
        } catch (error) {
          // Pause or stop requested - throw to abort AI SDK generation
          this.logger({
            category: "agent",
            message: `Agent paused/stopped: ${error.message}`,
            level: 1,
          });
          throw error;
        }
      }
      // AUDITARIA_FEATURE_END

      if (event.toolCalls && event.toolCalls.length > 0) {
        for (let i = 0; i < event.toolCalls.length; i++) {
          const toolCall = event.toolCalls[i];
          const args = toolCall.input;
          const toolResult = event.toolResults?.[i];

          if (event.text && event.text.length > 0) {
            state.collectedReasoning.push(event.text);
            this.logger({
              category: "agent",
              message: `reasoning: ${event.text}`,
              level: 1,
            });
          }

          if (toolCall.toolName === "close") {
            state.completed = true;
            if (args?.taskComplete) {
              const closeReasoning = args.reasoning;
              const allReasoning = state.collectedReasoning.join(" ");
              state.finalMessage = closeReasoning
                ? `${allReasoning} ${closeReasoning}`.trim()
                : allReasoning || "Task completed successfully";
            }
          }
          const mappedActions = mapToolResultToActions({
            toolCallName: toolCall.toolName,
            toolResult,
            args,
            reasoning: event.text || undefined,
          });

          for (const action of mappedActions) {
            action.pageUrl = state.currentPageUrl;
            action.timestamp = Date.now();
            state.actions.push(action);
          }
        }
        state.currentPageUrl = (await this.v3.context.awaitActivePage()).url();

        // AUDITARIA_START: Emit "completed" update AFTER processing tool calls
        // This shows the final results and reasoning for the step
        if (onStep) {
          onStep({
            stepNumber,
            actions: state.actions.slice(-event.toolCalls.length),
            message: event.text || state.collectedReasoning.slice(-1)[0] || "",
            completed: true, // Step is now complete (always true here, even if task not done)
            currentUrl: state.currentPageUrl,
          });
        } // AUDITARIA_END
      }
    };
  }

  public async execute(
    instructionOrOptions: string | AgentExecuteOptions,
  ): Promise<AgentResult> {
    const startTime = Date.now();
    const {
      options, // AUDITARIA_MODIFY
      maxSteps,
      systemPrompt,
      allTools,
      messages,
      wrappedModel,
      initialPageUrl,
    } = await this.prepareAgent(instructionOrOptions);

    const state: AgentState = {
      collectedReasoning: [],
      actions: [],
      finalMessage: "",
      completed: false,
      currentPageUrl: initialPageUrl,
    };

    try {
      // AUDITARIA_START: Route to OAuth path if using CodeAssistClient
      if (this.isOAuthMode()) {
        return await this.executeWithCodeAssist(
          options,
          startTime,
          state,
          systemPrompt,
          allTools,
          messages,
          maxSteps,
        );
      }
      // AUDITARIA_END 

      // Existing AI SDK path (unchanged)
      const result = await this.llmClient.generateText({
        model: wrappedModel,
        system: systemPrompt,
        messages,
        tools: allTools,
        stopWhen: (result) => this.handleStop(result, maxSteps),
        temperature: 1,
        toolChoice: "auto",
        onStepFinish: this.createStepHandler(state, options.onStep, options.checkPauseState), // AUDITARIA_MODIFY: Enhanced step handler
      });

      return this.consolidateMetricsAndResult(startTime, state, result);
    } catch (error) {
      const errorMessage = error?.message ?? String(error);
      this.logger({
        category: "agent",
        message: `Error executing agent task: ${errorMessage}`,
        level: 0,
      });
      return {
        success: false,
        actions: state.actions,
        message: `Failed to execute task: ${errorMessage}`,
        completed: false,
      };
    }
  }

  /**
   * AUDITARIA_START: OAuth-compatible agent execution
   * Uses CodeAssistClient.createChatCompletion() directly instead of AI SDK
   * Manually implements agent loop with tool calling
   */
  private async executeWithCodeAssist(
    options: AgentExecuteOptions,
    startTime: number,
    state: AgentState,
    systemPrompt: string,
    allTools: ToolSet,
    messages: ModelMessage[],
    maxSteps: number,
  ): Promise<AgentResult> {
    this.logger({
      category: "agent",
      message: "Using OAuth mode (CodeAssist direct API calls)",
      level: 1,
    });

    // Convert messages to ChatMessage format
    const chatMessages: ChatMessage[] = [
      { role: "system", content: systemPrompt },
      ...messages.map((msg) => ({
        role: msg.role as "user" | "assistant" | "system",
        content:
          typeof msg.content === "string"
            ? msg.content
            : JSON.stringify(msg.content),
      })),
    ];

    let totalUsage = {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    };

    // Agent loop - manually execute tool calls
    let stepNumber = 0;
    for (let step = 0; step < maxSteps && !state.completed; step++) {
      // AUDITARIA_FEATURE: Check if execution should pause
      if (options.checkPauseState) {
        try {
          await options.checkPauseState();
        } catch (error) {
          // Stop requested
          this.logger({
            category: "agent",
            message: `Agent stopped: ${error.message}`,
            level: 1,
          });
          break;
        }
      }

      stepNumber++;

      // AUDITARIA: Emit "starting" update BEFORE API call
      // This allows UI to show which step is currently executing
      if (options.onStep) {
        options.onStep({
          stepNumber,
          actions: [],
          message: "", // No reasoning yet - step just started
          completed: false, // Signals step is starting/executing
          currentUrl: state.currentPageUrl,
        });
      }

      this.logger({
        category: "agent",
        message: `Step ${stepNumber}/${maxSteps}: Calling CodeAssist API`,
        level: 2,
      });

      // Call CodeAssist API with tools
      const response: LLMResponse =
        await this.llmClient.createChatCompletion({
          options: {
            messages: chatMessages,
            tools: this.convertToolsToLLMFormat(allTools),
            temperature: 1,
            tool_choice: "auto",
          },
          logger: this.logger,
        });

      // Accumulate usage
      if (response.usage) {
        totalUsage.prompt_tokens += response.usage.prompt_tokens || 0;
        totalUsage.completion_tokens += response.usage.completion_tokens || 0;
        totalUsage.total_tokens += response.usage.total_tokens || 0;
      }

      const choice = response.choices[0];
      if (!choice) {
        this.logger({
          category: "agent",
          message: "No response choice from LLM",
          level: 0,
        });
        break;
      }

      const toolCalls = choice.message?.tool_calls || [];
      const textContent = choice.message?.content || "";

      // Collect reasoning text
      if (textContent) {
        state.collectedReasoning.push(textContent);
        this.logger({
          category: "agent",
          message: `reasoning: ${textContent}`,
          level: 1,
        });
      }

      // If no tool calls, the agent is done
      if (toolCalls.length === 0) {
        this.logger({
          category: "agent",
          message: "No more tool calls, agent complete",
          level: 1,
        });
        break;
      }

      // AUDITARIA: Add assistant message with tool calls in proper format
      // Dont stringify - CodeAssistClient will convert to functionCall parts
      chatMessages.push({
        role: "assistant",
        content: textContent || "",  // Include reasoning text, not tool calls JSON
        tool_calls: toolCalls,        // Pass tool_calls separately for CodeAssistClient
      } as any);

      // Execute each tool call
      const toolResults: string[] = [];
      for (const toolCall of toolCalls) {
        const toolName = toolCall.function.name;
        const toolArgs = JSON.parse(toolCall.function.arguments);

        this.logger({
          category: "agent",
          message: `Executing tool: ${toolName}`,
          level: 1,
        });

        // Check for close tool
        if (toolName === "close") {
          state.completed = true;
          if (toolArgs?.taskComplete) {
            const closeReasoning = toolArgs.reasoning || "";
            const allReasoning = state.collectedReasoning.join(" ");
            state.finalMessage = closeReasoning
              ? `${allReasoning} ${closeReasoning}`.trim()
              : allReasoning || "Task completed successfully";
          } else {
            state.finalMessage = "Task could not be completed";
          }

          // Map to actions for tracking
          const mappedActions = mapToolResultToActions({
            toolCallName: toolName,
            toolResult: { success: true, reasoning: toolArgs.reasoning },
            args: toolArgs,
            reasoning: textContent || undefined,
          });

          for (const action of mappedActions) {
            action.pageUrl = state.currentPageUrl;
            action.timestamp = Date.now();
            state.actions.push(action);
          }

          // AUDITARIA: Emit "completed" update for close tool
          if (options.onStep) {
            options.onStep({
              stepNumber,
              actions: state.actions.slice(-mappedActions.length),
              message:
                textContent || state.collectedReasoning.slice(-1)[0] || "",
              completed: true, // Step is complete (close tool executed)
              currentUrl: state.currentPageUrl,
            });
          }

          toolResults.push(JSON.stringify({ success: true }));
          break; // Exit tool loop
        }

        // Execute the tool
        try {
          const tool = allTools[toolName];
          if (!tool) {
            this.logger({
              category: "agent",
              message: `Tool not found: ${toolName}`,
              level: 0,
            });
            toolResults.push(
              JSON.stringify({ error: `Unknown tool: ${toolName}` }),
            );
            continue;
          }

          const toolResult = await tool.execute(toolArgs, {});

          // Map tool result to actions
          const mappedActions = mapToolResultToActions({
            toolCallName: toolName,
            toolResult,
            args: toolArgs,
            reasoning: textContent || undefined,
          });

          for (const action of mappedActions) {
            action.pageUrl = state.currentPageUrl;
            action.timestamp = Date.now();
            state.actions.push(action);
          }

          // Update current URL after action
          state.currentPageUrl = (
            await this.v3.context.awaitActivePage()
          ).url();

          toolResults.push(JSON.stringify(toolResult));

          // AUDITARIA: Emit "completed" update AFTER tool execution
          if (options.onStep) {
            options.onStep({
              stepNumber,
              actions: state.actions.slice(-mappedActions.length),
              message:
                textContent || state.collectedReasoning.slice(-1)[0] || "",
              completed: true, // Step is complete (tool executed)
              currentUrl: state.currentPageUrl,
            });
          }
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : String(error);
          this.logger({
            category: "agent",
            message: `Tool execution error (${toolName}): ${errorMessage}`,
            level: 0,
          });
          toolResults.push(JSON.stringify({ error: errorMessage }));
        }
      }

      // Add tool results as user message
      chatMessages.push({
        role: "user",
        content: toolResults.join("\n"),
      });

      // If close tool was called, stop
      if (state.completed) {
        break;
      }
    }

    // Build final result
    const endTime = Date.now();
    const inferenceTimeMs = endTime - startTime;

    if (!state.finalMessage) {
      const allReasoning = state.collectedReasoning.join(" ").trim();
      state.finalMessage = allReasoning || "Task execution completed";
    }

    // Update metrics
    this.v3.updateMetrics(
      V3FunctionName.AGENT,
      totalUsage.prompt_tokens,
      totalUsage.completion_tokens,
      0, // reasoning tokens (not provided by Code Assist)
      0, // cached input tokens (not provided by Code Assist)
      inferenceTimeMs,
    );

    return {
      success: state.completed,
      message: state.finalMessage,
      actions: state.actions,
      completed: state.completed,
      usage: {
        input_tokens: totalUsage.prompt_tokens,
        output_tokens: totalUsage.completion_tokens,
        reasoning_tokens: 0,
        cached_input_tokens: 0,
        inference_time_ms: inferenceTimeMs,
      },
    };
  } // AUDITARIA_END

  public async stream(
    instructionOrOptions: string | AgentExecuteOptions,
  ): Promise<AgentStreamResult> {
    const {
      options, // AUDITARIA_MODIFY
      maxSteps,
      systemPrompt,
      allTools,
      messages,
      wrappedModel,
      initialPageUrl,
    } = await this.prepareAgent(instructionOrOptions);

    const state: AgentState = {
      collectedReasoning: [],
      actions: [],
      finalMessage: "",
      completed: false,
      currentPageUrl: initialPageUrl,
    };
    const startTime = Date.now();

    let resolveResult: (value: AgentResult | PromiseLike<AgentResult>) => void;
    let rejectResult: (reason: unknown) => void;
    const resultPromise = new Promise<AgentResult>((resolve, reject) => {
      resolveResult = resolve;
      rejectResult = reject;
    });

    const handleError = (error: unknown) => {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      this.logger({
        category: "agent",
        message: `Error during streaming: ${errorMessage}`,
        level: 0,
      });
      rejectResult(error);
    };

    const streamResult = this.llmClient.streamText({
      model: wrappedModel,
      system: systemPrompt,
      messages,
      tools: allTools,
      stopWhen: (result) => this.handleStop(result, maxSteps),
      temperature: 1,
      toolChoice: "auto",
      onStepFinish: this.createStepHandler(state, options.onStep, options.checkPauseState), // AUDITARIA_MODIFY: Enhanced step handler
      onError: ({ error }) => {
        handleError(error);
      },
      onFinish: (event) => {
        try {
          const result = this.consolidateMetricsAndResult(
            startTime,
            state,
            event,
          );
          resolveResult(result);
        } catch (error) {
          handleError(error);
        }
      },
    });

    const agentStreamResult = streamResult as AgentStreamResult;
    agentStreamResult.result = resultPromise;
    return agentStreamResult;
  }

  private consolidateMetricsAndResult(
    startTime: number,
    state: AgentState,
    result: { text?: string; usage?: LanguageModelUsage },
  ): AgentResult {
    if (!state.finalMessage) {
      const allReasoning = state.collectedReasoning.join(" ").trim();
      state.finalMessage = allReasoning || result.text || "";
    }

    const endTime = Date.now();
    const inferenceTimeMs = endTime - startTime;
    if (result.usage) {
      this.v3.updateMetrics(
        V3FunctionName.AGENT,
        result.usage.inputTokens || 0,
        result.usage.outputTokens || 0,
        result.usage.reasoningTokens || 0,
        result.usage.cachedInputTokens || 0,
        inferenceTimeMs,
      );
    }

    return {
      success: state.completed,
      message: state.finalMessage || "Task execution completed",
      actions: state.actions,
      completed: state.completed,
      usage: result.usage
        ? {
            input_tokens: result.usage.inputTokens || 0,
            output_tokens: result.usage.outputTokens || 0,
            reasoning_tokens: result.usage.reasoningTokens || 0,
            cached_input_tokens: result.usage.cachedInputTokens || 0,
            inference_time_ms: inferenceTimeMs,
          }
        : undefined,
    };
  }

  private buildSystemPrompt(
    executionInstruction: string,
    systemInstructions?: string,
  ): string {
    if (systemInstructions) {
      return `${systemInstructions}\nYour current goal: ${executionInstruction} when the task is complete, use the "close" tool with taskComplete: true`;
    }
    return `You are a web automation assistant using browser automation tools to accomplish the user's goal.\n\nYour task: ${executionInstruction}\n\nYou have access to various browser automation tools. Use them step by step to complete the task.\n\nIMPORTANT GUIDELINES:\n1. Always start by understanding the current page state\n2. Use the screenshot tool to verify page state when needed\n3. Use appropriate tools for each action\n4. When the task is complete, use the "close" tool with taskComplete: true\n5. If the task cannot be completed, use "close" with taskComplete: false\n\nTOOLS OVERVIEW:\n- screenshot: Take a PNG screenshot for quick visual context (use sparingly)\n- ariaTree: Get an accessibility (ARIA) hybrid tree for full page context\n- act: Perform a specific atomic action (click, type, etc.)\n- extract: Extract structured data\n- goto: Navigate to a URL\n- wait/navback/refresh: Control timing and navigation\n- scroll: Scroll the page x pixels up or down\n\nSTRATEGY:\n- Prefer ariaTree to understand the page before acting; use screenshot for confirmation.\n- Keep actions atomic and verify outcomes before proceeding.`;
  }

  private createTools() {
    return createAgentTools(this.v3, {
      executionModel: this.executionModel,
      logger: this.logger,
    });
  }

  /**
   * AUDITARIA_START: Convert AI SDK ToolSet format to LLM function calling format
   * Code Assist API uses OpenAI-compatible function calling format
   * CRITICAL: Code Assist API requires all function parameters to have "type": "object"
   *
   * NOTE: AI SDK tool() returns objects with inputSchema (Zod), NOT parameters.
   * We must convert the Zod schema to JSON Schema using zodToJsonSchema().
   */
  private convertToolsToLLMFormat(tools: ToolSet): LLMTool[] {
    return Object.entries(tools).map(([name, tool]) => {
      // Convert Zod inputSchema to JSON Schema
      // tool.inputSchema is the Zod schema, tool.parameters is undefined in AI SDK
      const jsonSchema = tool.inputSchema
        ? zodToJsonSchema(tool.inputSchema as Parameters<typeof zodToJsonSchema>[0], { target: 'openApi3' })
        : { type: 'object', properties: {}, additionalProperties: false };

      const finalParameters = {
        ...jsonSchema,
        type: "object", // Always ensure type is "object" for Code Assist API
      };

      return {
        type: "function" as const,
        name,
        description: tool.description || "",
        parameters: finalParameters as Record<string, unknown>,
      };
    });
  } // AUDITARIA_END

  private handleStop(
    result: Parameters<ReturnType<typeof stepCountIs>>[0],
    maxSteps: number,
  ): boolean | PromiseLike<boolean> {
    const lastStep = result.steps[result.steps.length - 1];
    if (lastStep?.toolCalls?.some((tc) => tc.toolName === "close")) {
      return true;
    }
    return stepCountIs(maxSteps)(result);
  }
}
