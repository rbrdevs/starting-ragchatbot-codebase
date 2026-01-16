"""
API Endpoint Tests for the RAG Chatbot System.

These tests use a test FastAPI app defined in conftest.py that mirrors
the production app but without static file mounting (which doesn't exist
in the test environment).
"""

import pytest
from unittest.mock import MagicMock


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_without_session_creates_new_session(self, client, mock_rag_system):
        """Query without session_id should create a new session"""
        response = client.post(
            "/api/query",
            json={"query": "What is machine learning?", "session_id": None}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_with_existing_session(self, client, mock_rag_system):
        """Query with existing session_id should use that session"""
        response = client.post(
            "/api/query",
            json={"query": "Tell me more", "session_id": "existing-session-456"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session-456"
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_query_returns_answer_and_sources(self, client, mock_rag_system):
        """Query should return both answer and sources"""
        response = client.post(
            "/api/query",
            json={"query": "Explain neural networks"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is a test answer about machine learning."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Test Course - Lesson 1"
        assert data["sources"][0]["link"] == "https://example.com/lesson1"

    def test_query_calls_rag_system_correctly(self, client, mock_rag_system):
        """Query should call RAG system with correct parameters"""
        client.post(
            "/api/query",
            json={"query": "What is deep learning?", "session_id": "session-789"}
        )

        mock_rag_system.query.assert_called_once_with(
            "What is deep learning?",
            "session-789"
        )

    def test_query_with_empty_query(self, client):
        """Empty query should still be processed (validation is not enforced)"""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )
        # Empty queries are allowed by the current implementation
        assert response.status_code == 200

    def test_query_missing_required_field(self, client):
        """Missing required 'query' field should return 422"""
        response = client.post(
            "/api/query",
            json={"session_id": "some-session"}
        )

        assert response.status_code == 422

    def test_query_invalid_json(self, client):
        """Invalid JSON should return 422"""
        response = client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_error_returns_500(self, client_with_errors):
        """Server error during query should return 500"""
        response = client_with_errors.post(
            "/api/query",
            json={"query": "This will fail"}
        )

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_get_courses_returns_stats(self, client, mock_rag_system):
        """GET /api/courses should return course statistics"""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "AI Fundamentals" in data["course_titles"]
        assert "Machine Learning Basics" in data["course_titles"]
        assert "Deep Learning Advanced" in data["course_titles"]

    def test_get_courses_calls_analytics(self, client, mock_rag_system):
        """GET /api/courses should call get_course_analytics"""
        client.get("/api/courses")

        mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_courses_error_returns_500(self, client_with_errors):
        """Server error during courses fetch should return 500"""
        response = client_with_errors.get("/api/courses")

        assert response.status_code == 500
        assert "Failed to retrieve analytics" in response.json()["detail"]


class TestRootEndpoint:
    """Tests for GET / endpoint (health check)"""

    def test_root_returns_status(self, client):
        """Root endpoint should return status ok"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data


class TestAsyncEndpoints:
    """Async tests for API endpoints using httpx"""

    @pytest.mark.asyncio
    async def test_async_query(self, async_client, mock_rag_system):
        """Test query endpoint with async client"""
        response = await async_client.post(
            "/api/query",
            json={"query": "Async query test"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    @pytest.mark.asyncio
    async def test_async_get_courses(self, async_client):
        """Test courses endpoint with async client"""
        response = await async_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data

    @pytest.mark.asyncio
    async def test_async_root(self, async_client):
        """Test root endpoint with async client"""
        response = await async_client.get("/")

        assert response.status_code == 200


class TestResponseModels:
    """Tests for response model validation"""

    def test_query_response_structure(self, client):
        """Query response should match expected structure"""
        response = client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        data = response.json()
        # Verify all required fields are present
        required_fields = {"answer", "sources", "session_id"}
        assert required_fields.issubset(data.keys())

        # Verify types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_courses_response_structure(self, client):
        """Courses response should match expected structure"""
        response = client.get("/api/courses")

        data = response.json()
        required_fields = {"total_courses", "course_titles"}
        assert required_fields.issubset(data.keys())

        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_query_with_special_characters(self, client):
        """Query with special characters should be handled"""
        response = client.post(
            "/api/query",
            json={"query": "What about C++ and @#$% symbols?"}
        )

        assert response.status_code == 200

    def test_query_with_unicode(self, client):
        """Query with unicode characters should be handled"""
        response = client.post(
            "/api/query",
            json={"query": "What about emojis? 🤖 and accents: café résumé"}
        )

        assert response.status_code == 200

    def test_query_with_very_long_text(self, client):
        """Very long query should be handled"""
        long_query = "What is " + "machine learning " * 100 + "?"
        response = client.post(
            "/api/query",
            json={"query": long_query}
        )

        assert response.status_code == 200

    def test_multiple_concurrent_queries(self, client):
        """Multiple queries should work correctly"""
        queries = [
            {"query": f"Query number {i}"} for i in range(5)
        ]

        responses = [
            client.post("/api/query", json=q)
            for q in queries
        ]

        assert all(r.status_code == 200 for r in responses)
