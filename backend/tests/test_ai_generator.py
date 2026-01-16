import pytest
from unittest.mock import MagicMock, patch
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test AIGenerator functionality"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create an AIGenerator instance with mock client"""
        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")
        generator.client = mock_anthropic_client
        return generator

    def test_initialization(self):
        """Test AIGenerator initialization"""
        with patch("ai_generator.anthropic.Anthropic"):
            generator = AIGenerator(api_key="test_key", model="test_model")

            assert generator.model == "test_model"
            assert generator.base_params["model"] == "test_model"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800

    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_client):
        """Test generating response without tool use"""
        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is a test response")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        result = ai_generator.generate_response(query="What is AI?")

        assert result == "This is a test response"
        assert mock_anthropic_client.messages.create.call_count == 1

        # Verify call parameters
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["messages"][0]["content"] == "What is AI?"
        assert call_args[1]["messages"][0]["role"] == "user"

    def test_generate_response_with_conversation_history(
        self, ai_generator, mock_anthropic_client
    ):
        """Test generating response with conversation history"""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        history = "User: What is AI?\nAssistant: AI is artificial intelligence."
        result = ai_generator.generate_response(
            query="Tell me more", conversation_history=history
        )

        # Verify history is in system prompt
        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" in system_content
        assert history in system_content

    def test_generate_response_with_tools_no_tool_use(
        self, ai_generator, mock_anthropic_client
    ):
        """Test response generation with tools available but not used"""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Direct response")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        tools = [
            {
                "name": "search_course_content",
                "description": "Search courses",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

        result = ai_generator.generate_response(query="Hello", tools=tools)

        assert result == "Direct response"

        # Verify tools were passed
        call_args = mock_anthropic_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}

    def test_generate_response_with_tool_use(self, ai_generator, mock_anthropic_client):
        """Test response generation when tool is used"""
        # First response: tool use
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "toolu_123"
        tool_use_block.name = "search_course_content"
        tool_use_block.input = {
            "query": "neural networks",
            "course_name": None,
            "lesson_number": None,
        }

        first_response = MagicMock()
        first_response.content = [tool_use_block]
        first_response.stop_reason = "tool_use"

        # Second response: final answer
        second_response = MagicMock()
        second_response.content = [
            MagicMock(text="Neural networks are fundamental to AI.")
        ]
        second_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            first_response,
            second_response,
        ]

        # Mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = (
            "[AI Course - Lesson 1]\nNeural networks content..."
        )

        tools = [{"name": "search_course_content", "description": "Search courses"}]

        result = ai_generator.generate_response(
            query="What are neural networks?",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        assert result == "Neural networks are fundamental to AI."
        assert mock_anthropic_client.messages.create.call_count == 2

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="neural networks",
            course_name=None,
            lesson_number=None,
        )

    def test_system_prompt_content(self, ai_generator):
        """Test that system prompt contains expected guidelines"""
        assert "course materials" in AIGenerator.SYSTEM_PROMPT
        assert "tool" in AIGenerator.SYSTEM_PROMPT.lower()
        assert "two sequential tool uses" in AIGenerator.SYSTEM_PROMPT.lower()

    def test_no_tool_manager_with_tool_use(self, ai_generator, mock_anthropic_client):
        """Test that tool use without tool manager just returns the response"""
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "toolu_123"

        response = MagicMock()
        response.content = [tool_use_block]
        response.stop_reason = "tool_use"

        mock_anthropic_client.messages.create.return_value = response

        tools = [{"name": "test_tool"}]

        # Without tool_manager, should not execute tool handling
        # This would be an error case, but we want to verify behavior
        result = ai_generator.generate_response(
            query="test", tools=tools, tool_manager=None
        )

        # Should not attempt second API call
        assert mock_anthropic_client.messages.create.call_count == 1

    def test_temperature_zero_for_determinism(
        self, ai_generator, mock_anthropic_client
    ):
        """Test that temperature is set to 0 for deterministic responses"""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        ai_generator.generate_response(query="test")

        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["temperature"] == 0

    def test_max_tokens_configuration(self, ai_generator, mock_anthropic_client):
        """Test that max_tokens is configured correctly"""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        ai_generator.generate_response(query="test")

        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["max_tokens"] == 800

    def test_two_sequential_tool_calls(self, ai_generator, mock_anthropic_client):
        """Test that Claude can make 2 sequential tool calls"""
        # Round 1: Tool use (get_course_outline)
        tool_use_1 = MagicMock()
        tool_use_1.type = "tool_use"
        tool_use_1.id = "toolu_1"
        tool_use_1.name = "get_course_outline"
        tool_use_1.input = {"course_name": "AI Course"}

        response_1 = MagicMock()
        response_1.content = [tool_use_1]
        response_1.stop_reason = "tool_use"

        # Round 2: Tool use (search_course_content) after seeing outline
        tool_use_2 = MagicMock()
        tool_use_2.type = "tool_use"
        tool_use_2.id = "toolu_2"
        tool_use_2.name = "search_course_content"
        tool_use_2.input = {"query": "neural networks", "lesson_number": 4}

        response_2 = MagicMock()
        response_2.content = [tool_use_2]
        response_2.stop_reason = "tool_use"

        # Final: Text response after 2 tool rounds
        text_block = MagicMock()
        text_block.text = (
            "Based on the outline and content, lesson 4 covers neural networks."
        )

        response_3 = MagicMock()
        response_3.content = [text_block]
        response_3.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            response_1,
            response_2,
            response_3,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = [
            "Outline: Lesson 4 - Neural Networks",  # Round 1 result
            "Content about neural networks...",  # Round 2 result
        ]

        tools = [
            {"name": "get_course_outline", "description": "Get course outline"},
            {"name": "search_course_content", "description": "Search content"},
        ]

        result = ai_generator.generate_response(
            query="What's in lesson 4 of AI course?",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Assertions
        assert (
            result
            == "Based on the outline and content, lesson 4 covers neural networks."
        )
        assert mock_anthropic_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify message accumulation - final call should have 5 messages
        final_call_messages = mock_anthropic_client.messages.create.call_args_list[2][
            1
        ]["messages"]
        assert len(final_call_messages) == 5  # [user, asst, user, asst, user]

        # Verify first call has 1 message (user query)
        first_call_messages = mock_anthropic_client.messages.create.call_args_list[0][
            1
        ]["messages"]
        assert len(first_call_messages) == 1

        # Verify second call has 3 messages (user, asst with tool use, user with tool results)
        second_call_messages = mock_anthropic_client.messages.create.call_args_list[1][
            1
        ]["messages"]
        assert len(second_call_messages) == 3

    def test_max_rounds_termination(self, ai_generator, mock_anthropic_client):
        """Test that loop terminates after 2 rounds even if Claude wants more tools"""
        # Setup: Always return tool_use to simulate Claude wanting infinite tools
        tool_use = MagicMock()
        tool_use.type = "tool_use"
        tool_use.id = "toolu_infinite"
        tool_use.name = "search_course_content"
        tool_use.input = {"query": "test"}

        response = MagicMock()
        response.content = [tool_use]
        response.stop_reason = "tool_use"

        # Return tool_use for all calls
        mock_anthropic_client.messages.create.return_value = response

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Result"

        tools = [{"name": "search_course_content"}]

        result = ai_generator.generate_response(
            query="test query", tools=tools, tool_manager=mock_tool_manager
        )

        # Should stop after 2 rounds (2 API calls) + 1 final call without tools (3 total)
        assert mock_anthropic_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2
        assert "Maximum tool usage rounds reached" in result

    def test_early_termination_no_tool_use(self, ai_generator, mock_anthropic_client):
        """Test that loop exits early if Claude doesn't use tools"""
        text_block = MagicMock()
        text_block.text = "Direct answer without tools"

        response = MagicMock()
        response.content = [text_block]
        response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.return_value = response

        mock_tool_manager = MagicMock()
        tools = [{"name": "search_course_content"}]

        result = ai_generator.generate_response(
            query="Hello", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "Direct answer without tools"
        assert mock_anthropic_client.messages.create.call_count == 1
        # Tool manager should never be called
        assert mock_tool_manager.execute_tool.call_count == 0

    def test_early_termination_after_round_1(self, ai_generator, mock_anthropic_client):
        """Test that loop exits after round 1 if Claude doesn't use tools in round 2"""
        # Round 1: Tool use
        tool_use = MagicMock()
        tool_use.type = "tool_use"
        tool_use.id = "toolu_1"
        tool_use.name = "search_course_content"
        tool_use.input = {"query": "test"}

        response_1 = MagicMock()
        response_1.content = [tool_use]
        response_1.stop_reason = "tool_use"

        # Round 2: Direct text (no tool use)
        text_block = MagicMock()
        text_block.text = "Here's the answer based on search results"

        response_2 = MagicMock()
        response_2.content = [text_block]
        response_2.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [response_1, response_2]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Search results..."

        tools = [{"name": "search_course_content"}]

        result = ai_generator.generate_response(
            query="test", tools=tools, tool_manager=mock_tool_manager
        )

        # Should make 2 API calls (round 1 + round 2)
        assert mock_anthropic_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1
        assert result == "Here's the answer based on search results"

    def test_tool_execution_error_handling(self, ai_generator, mock_anthropic_client):
        """Test graceful handling of tool execution errors"""
        # Round 1: Tool use
        tool_use = MagicMock()
        tool_use.type = "tool_use"
        tool_use.id = "toolu_err"
        tool_use.name = "search_course_content"
        tool_use.input = {"query": "test"}

        response_1 = MagicMock()
        response_1.content = [tool_use]
        response_1.stop_reason = "tool_use"

        # Final: Claude's response to error
        text_block = MagicMock()
        text_block.text = "I encountered an error while searching the content."

        response_2 = MagicMock()
        response_2.content = [text_block]
        response_2.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [response_1, response_2]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = Exception("DB connection failed")

        tools = [{"name": "search_course_content"}]

        result = ai_generator.generate_response(
            query="Search for something", tools=tools, tool_manager=mock_tool_manager
        )

        # Should make 2 API calls: 1 for tool use, 1 for Claude to handle error
        assert mock_anthropic_client.messages.create.call_count == 2
        assert "I encountered an error" in result

        # Verify error message was passed to Claude
        second_call_messages = mock_anthropic_client.messages.create.call_args_list[1][
            1
        ]["messages"]
        tool_results = second_call_messages[2]["content"]
        assert "Error executing tool" in tool_results[0]["content"]
        assert tool_results[0]["is_error"] is True

    def test_message_list_accumulation(self, ai_generator, mock_anthropic_client):
        """Test that messages list grows correctly through rounds"""
        # Setup responses for 2 tool rounds
        tool_use_1 = MagicMock()
        tool_use_1.type = "tool_use"
        tool_use_1.id = "toolu_1"
        tool_use_1.name = "search_course_content"
        tool_use_1.input = {"query": "test1"}

        response_1 = MagicMock()
        response_1.content = [tool_use_1]
        response_1.stop_reason = "tool_use"

        tool_use_2 = MagicMock()
        tool_use_2.type = "tool_use"
        tool_use_2.id = "toolu_2"
        tool_use_2.name = "search_course_content"
        tool_use_2.input = {"query": "test2"}

        response_2 = MagicMock()
        response_2.content = [tool_use_2]
        response_2.stop_reason = "tool_use"

        text_block = MagicMock()
        text_block.text = "Final answer combining both searches"

        response_3 = MagicMock()
        response_3.content = [text_block]
        response_3.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            response_1,
            response_2,
            response_3,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        tools = [{"name": "search_course_content"}]

        ai_generator.generate_response(
            query="Compare test1 and test2", tools=tools, tool_manager=mock_tool_manager
        )

        # Check messages structure in each call
        call_1_messages = mock_anthropic_client.messages.create.call_args_list[0][1][
            "messages"
        ]
        assert len(call_1_messages) == 1  # [user]
        assert call_1_messages[0]["role"] == "user"

        call_2_messages = mock_anthropic_client.messages.create.call_args_list[1][1][
            "messages"
        ]
        assert len(call_2_messages) == 3  # [user, asst, user]
        assert call_2_messages[0]["role"] == "user"
        assert call_2_messages[1]["role"] == "assistant"
        assert call_2_messages[2]["role"] == "user"

        call_3_messages = mock_anthropic_client.messages.create.call_args_list[2][1][
            "messages"
        ]
        assert len(call_3_messages) == 5  # [user, asst, user, asst, user]
        assert call_3_messages[0]["role"] == "user"
        assert call_3_messages[1]["role"] == "assistant"
        assert call_3_messages[2]["role"] == "user"
        assert call_3_messages[3]["role"] == "assistant"
        assert call_3_messages[4]["role"] == "user"

    def test_system_prompt_updated_for_multi_round(self, ai_generator):
        """Test that system prompt contains multi-round guidance"""
        assert "two sequential tool uses" in AIGenerator.SYSTEM_PROMPT.lower()
        assert "step-by-step" in AIGenerator.SYSTEM_PROMPT.lower()
