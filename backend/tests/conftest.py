import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config

# API testing imports
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import Optional
import httpx


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


# ============================================================================
# API Testing Fixtures
# ============================================================================

# Pydantic models for API testing (mirrors app.py models)
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]


@pytest.fixture
def mock_rag_system():
    """Create a mock RAGSystem for API testing"""
    mock = MagicMock()

    # Mock session manager
    mock.session_manager = MagicMock()
    mock.session_manager.create_session.return_value = "test-session-123"

    # Mock query method
    mock.query.return_value = (
        "This is a test answer about machine learning.",
        [{"text": "Test Course - Lesson 1", "link": "https://example.com/lesson1"}]
    )

    # Mock get_course_analytics
    mock.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["AI Fundamentals", "Machine Learning Basics", "Deep Learning Advanced"]
    }

    return mock


@pytest.fixture
def mock_rag_system_error():
    """Create a mock RAGSystem that raises errors"""
    mock = MagicMock()
    mock.session_manager = MagicMock()
    mock.session_manager.create_session.return_value = "error-session"
    mock.query.side_effect = Exception("Database connection failed")
    mock.get_course_analytics.side_effect = Exception("Failed to retrieve analytics")
    return mock


def create_test_app(mock_rag_system):
    """
    Create a test FastAPI app without static file mounting.
    This avoids import issues since static files don't exist in test environment.
    """
    app = FastAPI(title="Course Materials RAG System - Test")

    # Store the mock in app state for access in endpoints
    app.state.rag_system = mock_rag_system

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            rag = app.state.rag_system
            session_id = request.session_id
            if not session_id:
                session_id = rag.session_manager.create_session()

            answer, sources = rag.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            rag = app.state.rag_system
            analytics = rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Root endpoint for health check"""
        return {"status": "ok", "message": "Course Materials RAG System"}

    return app


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with mocked RAG system"""
    return create_test_app(mock_rag_system)


@pytest.fixture
def test_app_with_errors(mock_rag_system_error):
    """Create a test FastAPI app with RAG system that raises errors"""
    return create_test_app(mock_rag_system_error)


@pytest.fixture
def client(test_app):
    """Create a synchronous test client"""
    return TestClient(test_app)


@pytest.fixture
def client_with_errors(test_app_with_errors):
    """Create a test client with error-raising RAG system"""
    return TestClient(test_app_with_errors)


@pytest.fixture
async def async_client(test_app):
    """Create an async test client using httpx"""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://testserver"
    ) as client:
        yield client


@pytest.fixture
def sample_query_request():
    """Sample query request data"""
    return {
        "query": "What is machine learning?",
        "session_id": None
    }


@pytest.fixture
def sample_query_request_with_session():
    """Sample query request with existing session"""
    return {
        "query": "Tell me more about neural networks",
        "session_id": "existing-session-456"
    }
