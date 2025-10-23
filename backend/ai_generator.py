import anthropic
from typing import List, Optional, Dict, Any
from config import config

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for retrieving course information.

Tool Usage Guidelines:
- **Course Outline Tool** (`get_course_outline`): Use for questions about course structure, outline, or lesson list
  - Returns: Course title, course link, and complete list of lessons (number and title)
  - Use when users ask: "What's the outline of...", "What lessons are in...", "Course structure", etc.

- **Content Search Tool** (`search_course_content`): Use for questions about specific course content or detailed materials
  - Returns: Relevant content chunks from course materials
  - Use when users ask about specific topics, concepts, or lesson details

- **Up to two sequential tool uses per query** - Use tools to gather information step-by-step
  - Example: First get course outline, then search for specific content from that outline
  - Example: Search for topic A, then search for topic B to compare them
  - Each tool call will be processed before you can make another
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use outline tool first, then provide the course structure
- **Course content questions**: Use search tool first, then answer based on retrieved content
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool usage explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the outline"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def _build_system_content(self, conversation_history: Optional[str] = None) -> str:
        """
        Build system content with optional conversation history.

        Args:
            conversation_history: Previous messages for context

        Returns:
            System content string
        """
        if conversation_history:
            return f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
        return self.SYSTEM_PROMPT

    def _execute_tool_loop(self, messages: List[Dict[str, Any]],
                           system_content: str,
                           max_rounds: int,
                           tools: Optional[List],
                           tool_manager) -> str:
        """
        Execute tool calling loop with support for sequential rounds.

        Args:
            messages: Initial messages list (user query)
            system_content: System prompt with optional history
            max_rounds: Maximum number of tool calling rounds
            tools: Available tool definitions
            tool_manager: Manager to execute tools

        Returns:
            Final response text
        """
        # Create a copy of messages to avoid mutating the original list
        messages = messages.copy()
        round_counter = 0
        current_response = None

        while round_counter < max_rounds:
            # 1. Make API call with tools (if available)
            # Pass a copy of messages to prevent later modifications from affecting the captured call args
            api_params = {
                **self.base_params,
                "messages": messages.copy(),
                "system": system_content
            }

            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**api_params)
            current_response = response

            # 2. Check stop reason - terminate if no tool use
            if response.stop_reason == "end_turn":
                break  # Claude decided not to use tools

            if response.stop_reason != "tool_use":
                # Handle unexpected stop reason
                break

            # 3. Check if tool_manager is available
            if not tool_manager:
                # Tool use requested but no manager available
                break

            # 4. Append assistant response to messages
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            # 5. Execute all tool calls and collect results
            tool_results = []
            tool_execution_failed = False

            for content_block in response.content:
                if content_block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(
                            content_block.name,
                            **content_block.input
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": result
                        })
                    except Exception as e:
                        # Handle tool execution error
                        tool_execution_failed = True
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Error executing tool: {str(e)}",
                            "is_error": True
                        })

            # 6. Append tool results to messages
            if tool_results:
                messages.append({
                    "role": "user",
                    "content": tool_results
                })

            # 7. If tool execution failed, make one final call for Claude to handle error
            if tool_execution_failed:
                final_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": system_content
                }
                final_response = self.client.messages.create(**final_params)
                return final_response.content[0].text

            # 8. Increment round counter
            round_counter += 1

        # After loop completes, check if we need to make a final call
        # If we exited because of max rounds and last response was tool_use, make final call
        if round_counter >= max_rounds and current_response and current_response.stop_reason == "tool_use":
            # Make one final API call without tools to get text response
            final_params = {
                **self.base_params,
                "messages": messages.copy(),
                "system": system_content
            }
            current_response = self.client.messages.create(**final_params)

        # Extract final text response
        if current_response and current_response.content:
            for block in current_response.content:
                if hasattr(block, 'text') and isinstance(block.text, str):
                    return block.text
            # If no text block found
            return "Maximum tool usage rounds reached. Please rephrase your question."

        return "No response generated"

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 sequential tool calling rounds.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content with conversation history
        system_content = self._build_system_content(conversation_history)

        # Build initial messages list
        messages = [{"role": "user", "content": query}]

        # Execute tool loop with configured max rounds
        return self._execute_tool_loop(
            messages=messages,
            system_content=system_content,
            max_rounds=config.MAX_TOOL_ROUNDS,
            tools=tools,
            tool_manager=tool_manager
        )