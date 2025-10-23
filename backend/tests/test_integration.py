import pytest
import os
import shutil
from unittest.mock import MagicMock, patch
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager
from models import Course, Lesson, CourseChunk
from rag_system import RAGSystem
from config import Config


@pytest.fixture(scope="module")
def test_db_path():
    """Provide a test database path"""
    path = "./test_integration_chroma_db"
    yield path
    # Cleanup after all tests
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.fixture
def clean_vector_store(test_db_path):
    """Create a clean VectorStore instance for each test"""
    # Clean up before test
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)

    store = VectorStore(
        chroma_path=test_db_path,
        embedding_model="all-MiniLM-L6-v2",
        max_results=5
    )
    yield store

    # Clean up after test
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)


@pytest.fixture
def populated_vector_store(clean_vector_store, sample_course, sample_course_chunks):
    """Create a VectorStore with sample data"""
    clean_vector_store.add_course_metadata(sample_course)
    clean_vector_store.add_course_content(sample_course_chunks)
    return clean_vector_store


class TestVectorStoreIntegration:
    """Integration tests with real ChromaDB"""

    def test_add_and_search_content(self, clean_vector_store, sample_course_chunks):
        """Test adding content and searching it"""
        # Add chunks
        clean_vector_store.add_course_content(sample_course_chunks)

        # Search for content
        results = clean_vector_store.search(query="neural networks")

        assert not results.is_empty()
        assert results.error is None
        assert len(results.documents) > 0

        # Verify relevant content is returned
        found_neural = any("neural" in doc.lower() or "network" in doc.lower()
                          for doc in results.documents)
        assert found_neural, "Should find content about neural networks"

    def test_add_course_metadata_and_resolve_name(self, clean_vector_store, sample_course):
        """Test adding course metadata and resolving course names"""
        clean_vector_store.add_course_metadata(sample_course)

        # Test exact match
        resolved = clean_vector_store._resolve_course_name("Test Course on AI")
        assert resolved == "Test Course on AI"

        # Test fuzzy match
        resolved = clean_vector_store._resolve_course_name("AI course")
        assert resolved == "Test Course on AI"

        # Test partial match
        resolved = clean_vector_store._resolve_course_name("Test")
        assert resolved == "Test Course on AI"

    def test_search_with_course_filter(self, populated_vector_store):
        """Test searching with course name filter"""
        results = populated_vector_store.search(
            query="machine learning",
            course_name="AI"  # Fuzzy match
        )

        # All results should be from the specified course
        for meta in results.metadata:
            assert meta["course_title"] == "Test Course on AI"

    def test_search_with_lesson_filter(self, populated_vector_store):
        """Test searching with lesson number filter"""
        results = populated_vector_store.search(
            query="content",
            lesson_number=1
        )

        # All results should be from lesson 1
        for meta in results.metadata:
            assert meta["lesson_number"] == 1

    def test_search_with_combined_filters(self, populated_vector_store):
        """Test searching with both course and lesson filters"""
        results = populated_vector_store.search(
            query="content",
            course_name="Test Course on AI",
            lesson_number=0
        )

        # All results should match both filters
        for meta in results.metadata:
            assert meta["course_title"] == "Test Course on AI"
            assert meta["lesson_number"] == 0

    def test_get_existing_course_titles(self, populated_vector_store, sample_course):
        """Test retrieving all course titles"""
        titles = populated_vector_store.get_existing_course_titles()

        assert len(titles) >= 1
        assert sample_course.title in titles

    def test_get_lesson_link(self, populated_vector_store, sample_course):
        """Test retrieving lesson links"""
        link = populated_vector_store.get_lesson_link("Test Course on AI", 0)

        assert link == sample_course.lessons[0].lesson_link

    def test_empty_search_results(self, populated_vector_store):
        """Test search with query that has no matches"""
        results = populated_vector_store.search(
            query="nonexistent quantum cryptography topic xyz"
        )

        # ChromaDB may still return some results due to semantic search
        # but they should have low relevance (high distance)
        if not results.is_empty():
            # If results exist, they should have relatively high distances
            assert all(d > 0.5 for d in results.distances), \
                "Results for irrelevant query should have high distance scores"


class TestCourseSearchToolIntegration:
    """Integration tests for CourseSearchTool with real VectorStore"""

    def test_execute_with_real_search(self, populated_vector_store):
        """Test CourseSearchTool.execute with real vector store"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute(query="neural networks")

        # Should return formatted results
        assert isinstance(result, str)
        assert "[Test Course on AI" in result
        assert "Lesson" in result

        # Should track sources
        assert len(tool.last_sources) > 0
        assert "text" in tool.last_sources[0]
        assert "link" in tool.last_sources[0]

    def test_execute_with_course_filter(self, populated_vector_store):
        """Test tool execution with course filter"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute(
            query="content",
            course_name="AI"  # Fuzzy match
        )

        assert "[Test Course on AI" in result
        assert result != "No relevant content found"

    def test_execute_with_invalid_course(self, populated_vector_store):
        """Test tool execution with invalid course name"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute(
            query="test",
            course_name="Nonexistent Course XYZ"
        )

        assert "No course found matching" in result or "No relevant content found" in result

    def test_source_tracking_with_links(self, populated_vector_store):
        """Test that sources include correct links"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute(query="neural networks", lesson_number=1)

        # Verify sources have links
        assert len(tool.last_sources) > 0
        for source in tool.last_sources:
            if source["link"]:
                assert source["link"].startswith("http")


class TestRAGSystemIntegration:
    """Integration tests for complete RAG system (with mocked AI)"""

    @pytest.fixture
    def integration_config(self, test_db_path):
        """Create config for integration testing"""
        return Config(
            ANTHROPIC_API_KEY="test_key",
            ANTHROPIC_MODEL="test_model",
            EMBEDDING_MODEL="all-MiniLM-L6-v2",
            CHUNK_SIZE=800,
            CHUNK_OVERLAP=100,
            MAX_RESULTS=5,
            MAX_HISTORY=2,
            CHROMA_PATH=test_db_path
        )

    @pytest.fixture
    def mock_ai_generator(self):
        """Mock AIGenerator to avoid API calls"""
        with patch('rag_system.AIGenerator') as mock_class:
            mock_generator = MagicMock()
            mock_class.return_value = mock_generator
            yield mock_generator

    def test_query_with_real_search(self, integration_config, mock_ai_generator, sample_course, sample_course_chunks):
        """Test RAG query with real vector search but mocked AI"""
        # Clean up test db
        if os.path.exists(integration_config.CHROMA_PATH):
            shutil.rmtree(integration_config.CHROMA_PATH)

        # Mock AI to trigger tool use
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.id = "toolu_123"
        tool_use_block.name = "search_course_content"
        tool_use_block.input = {"query": "neural networks", "course_name": None, "lesson_number": None}

        first_response = MagicMock()
        first_response.content = [tool_use_block]
        first_response.stop_reason = "tool_use"

        second_response = MagicMock()
        second_response.content = [MagicMock(text="Neural networks are fundamental.")]
        second_response.stop_reason = "end_turn"

        mock_ai_generator.generate_response.side_effect = lambda **kwargs: (
            "Neural networks are fundamental." if "tool_manager" in kwargs else "Direct response"
        )

        # Create RAG system
        rag_system = RAGSystem(integration_config)

        # Add course data
        rag_system.vector_store.add_course_metadata(sample_course)
        rag_system.vector_store.add_course_content(sample_course_chunks)

        # Execute query
        response, sources = rag_system.query("What are neural networks?")

        # Verify response
        assert response is not None
        assert isinstance(response, str)

        # Clean up
        if os.path.exists(integration_config.CHROMA_PATH):
            shutil.rmtree(integration_config.CHROMA_PATH)

    def test_end_to_end_document_processing(self, integration_config, mock_ai_generator):
        """Test adding a document and querying it"""
        # Clean up test db
        if os.path.exists(integration_config.CHROMA_PATH):
            shutil.rmtree(integration_config.CHROMA_PATH)

        mock_ai_generator.generate_response.return_value = "This is a test response."

        rag_system = RAGSystem(integration_config)

        # Create a test document
        test_doc_path = os.path.join(integration_config.CHROMA_PATH, "test_course.txt")
        os.makedirs(os.path.dirname(test_doc_path), exist_ok=True)

        with open(test_doc_path, 'w', encoding='utf-8') as f:
            f.write("""Course Title: Test Integration Course
Course Link: https://example.com/course
Course Instructor: Test Instructor

Lesson 0: Introduction
Lesson Link: https://example.com/lesson0
This is the introduction to our test course about integration testing.

Lesson 1: Advanced Topics
Lesson Link: https://example.com/lesson1
This lesson covers neural networks and deep learning concepts.
""")

        # Add the document
        course, num_chunks = rag_system.add_course_document(test_doc_path)

        assert course is not None
        assert course.title == "Test Integration Course"
        assert num_chunks > 0

        # Query the content
        response, sources = rag_system.query("Tell me about neural networks")

        assert response == "This is a test response."
        assert mock_ai_generator.generate_response.called

        # Clean up
        if os.path.exists(integration_config.CHROMA_PATH):
            shutil.rmtree(integration_config.CHROMA_PATH)


class TestToolManagerIntegration:
    """Integration tests for ToolManager with real tools"""

    def test_full_tool_workflow(self, populated_vector_store):
        """Test complete workflow of registering and executing tools"""
        # Create tool manager
        manager = ToolManager()

        # Register search tool
        search_tool = CourseSearchTool(populated_vector_store)
        manager.register_tool(search_tool)

        # Get tool definitions
        definitions = manager.get_tool_definitions()
        assert len(definitions) >= 1

        # Execute tool
        result = manager.execute_tool(
            "search_course_content",
            query="neural networks",
            course_name=None,
            lesson_number=None
        )

        assert isinstance(result, str)
        assert "Test Course on AI" in result or "No relevant content found" in result

        # Get sources
        sources = manager.get_last_sources()
        assert isinstance(sources, list)

        # Reset sources
        manager.reset_sources()
        sources_after_reset = manager.get_last_sources()
        assert sources_after_reset == []
