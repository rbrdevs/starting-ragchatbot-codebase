import pytest
from unittest.mock import MagicMock, patch
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test AIGenerator functionality"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
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
        with patch('ai_generator.anthropic.Anthropic'):
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

    def test_generate_response_with_conversation_history(self, ai_generator, mock_anthropic_client):
        """Test generating response with conversation history"""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        history = "User: What is AI?\nAssistant: AI is artificial intelligence."
        result = ai_generator.generate_response(query="Tell me more", conversation_history=history)

        # Verify history is in system prompt
        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" in system_content
        assert history in system_content

    def test_generate_response_with_tools_no_tool_use(self, ai_generator, mock_anthropic_client):
        """Test response generation with tools available but not used"""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Direct response")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        tools = [{
            "name": "search_course_content",
            "description": "Search courses",
            "input_schema": {"type": "object", "properties": {}}
        }]

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
        tool_use_block.input = {"query": "neural networks", "course_name": None, "lesson_number": None}

        first_response = MagicMock()
        first_response.content = [tool_use_block]
        first_response.stop_reason = "tool_use"

        # Second response: final answer
        second_response = MagicMock()
        second_response.content = [MagicMock(text="Neural networks are fundamental to AI.")]
        second_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        # Mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "[AI Course - Lesson 1]\nNeural networks content..."

        tools = [{
            "name": "search_course_content",
            "description": "Search courses"
        }]

        result = ai_generator.generate_response(
            query="What are neural networks?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        assert result == "Neural networks are fundamental to AI."
        assert mock_anthropic_client.messages.create.call_count == 2

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="neural networks",
            course_name=None,
            lesson_number=None
        )

    def test_handle_tool_execution_flow(self, ai_generator, mock_anthropic_client):
        """Test the tool execution flow in detail"""
        # Setup tool use response
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "toolu_456"
        tool_use_block.name = "search_course_content"
        tool_use_block.input = {"query": "test query"}

        initial_response = MagicMock()
        initial_response.content = [tool_use_block]

        # Setup final response
        final_response = MagicMock()
        final_response.content = [MagicMock(text="Final answer")]

        mock_anthropic_client.messages.create.return_value = final_response

        # Mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool result content"

        base_params = {
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "System prompt"
        }

        result = ai_generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)

        assert result == "Final answer"

        # Verify second API call structure
        call_args = mock_anthropic_client.messages.create.call_args
        messages = call_args[1]["messages"]

        # Should have 3 messages: original user, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Verify tool result structure
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "toolu_456"
        assert tool_results[0]["content"] == "Tool result content"

    def test_handle_multiple_tool_calls(self, ai_generator, mock_anthropic_client):
        """Test handling multiple tool calls in one response"""
        # Setup multiple tool use blocks
        tool_use_1 = MagicMock()
        tool_use_1.type = "tool_use"
        tool_use_1.id = "toolu_1"
        tool_use_1.name = "search_course_content"
        tool_use_1.input = {"query": "query1"}

        tool_use_2 = MagicMock()
        tool_use_2.type = "tool_use"
        tool_use_2.id = "toolu_2"
        tool_use_2.name = "search_course_content"
        tool_use_2.input = {"query": "query2"}

        initial_response = MagicMock()
        initial_response.content = [tool_use_1, tool_use_2]

        final_response = MagicMock()
        final_response.content = [MagicMock(text="Combined answer")]

        mock_anthropic_client.messages.create.return_value = final_response

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        base_params = {
            "messages": [{"role": "user", "content": "Test"}],
            "system": "System"
        }

        result = ai_generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify tool results in final call
        call_args = mock_anthropic_client.messages.create.call_args
        tool_results = call_args[1]["messages"][2]["content"]
        assert len(tool_results) == 2

    def test_system_prompt_content(self, ai_generator):
        """Test that system prompt contains expected guidelines"""
        assert "course materials" in AIGenerator.SYSTEM_PROMPT
        assert "tool" in AIGenerator.SYSTEM_PROMPT.lower()
        assert "one tool use per query maximum" in AIGenerator.SYSTEM_PROMPT.lower()

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
        result = ai_generator.generate_response(query="test", tools=tools, tool_manager=None)

        # Should not attempt second API call
        assert mock_anthropic_client.messages.create.call_count == 1

    def test_temperature_zero_for_determinism(self, ai_generator, mock_anthropic_client):
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
