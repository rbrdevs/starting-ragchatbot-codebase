import pytest
import sys
import os
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Test Course on AI",
        course_link="https://example.com/course",
        instructor="John Doe",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction",
                lesson_link="https://example.com/lesson0",
            ),
            Lesson(
                lesson_number=1,
                title="Basics",
                lesson_link="https://example.com/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Advanced Topics",
                lesson_link="https://example.com/lesson2",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Lesson 0 content: This is an introduction to AI and machine learning.",
            course_title="Test Course on AI",
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="We will cover basic concepts and terminology in this course.",
            course_title="Test Course on AI",
            lesson_number=0,
            chunk_index=1,
        ),
        CourseChunk(
            content="Lesson 1 content: Neural networks are the foundation of deep learning.",
            course_title="Test Course on AI",
            lesson_number=1,
            chunk_index=2,
        ),
        CourseChunk(
            content="Training involves adjusting weights to minimize loss functions.",
            course_title="Test Course on AI",
            lesson_number=1,
            chunk_index=3,
        ),
        CourseChunk(
            content="Lesson 2 content: Advanced topics include transformers and attention mechanisms.",
            course_title="Test Course on AI",
            lesson_number=2,
            chunk_index=4,
        ),
    ]


@pytest.fixture
def sample_search_results(sample_course_chunks):
    """Create sample search results"""
    chunks = sample_course_chunks[:3]  # Use first 3 chunks
    return SearchResults(
        documents=[chunk.content for chunk in chunks],
        metadata=[
            {
                "course_title": chunk.course_title,
                "lesson_number": chunk.lesson_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ],
        distances=[0.1, 0.2, 0.3],
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """Create search results with error"""
    return SearchResults.empty("Search error: Database connection failed")


@pytest.fixture
def mock_vector_store(sample_search_results):
    """Create a mock VectorStore"""
    mock = MagicMock()
    mock.search.return_value = sample_search_results
    mock.get_lesson_link.return_value = "https://example.com/lesson0"
    mock._resolve_course_name.return_value = "Test Course on AI"
    return mock


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client"""
    mock_client = MagicMock()

    # Mock response without tool use
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="This is a test response")]
    mock_response.stop_reason = "end_turn"

    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client_with_tool_use():
    """Create a mock Anthropic client that triggers tool use"""
    mock_client = MagicMock()

    # First response: Tool use
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

    # Second response: Final answer
    second_response = MagicMock()
    second_response.content = [
        MagicMock(text="Neural networks are the foundation of deep learning.")
    ]
    second_response.stop_reason = "end_turn"

    # Configure mock to return different responses
    mock_client.messages.create.side_effect = [first_response, second_response]

    return mock_client


@pytest.fixture
def test_config():
    """Create a test configuration"""
    return Config(
        ANTHROPIC_API_KEY="test_api_key",
        ANTHROPIC_MODEL="claude-sonnet-4-20250514",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        CHROMA_PATH="./test_chroma_db",
    )


@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager"""
    mock = MagicMock()
    mock.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                    "lesson_number": {"type": "integer"},
                },
                "required": ["query"],
            },
        }
    ]
    mock.execute_tool.return_value = "[Test Course - Lesson 0]\nThis is test content."
    mock.get_last_sources.return_value = [
        {"text": "Test Course - Lesson 0", "link": "https://example.com/lesson0"}
    ]
    return mock
