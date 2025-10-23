import pytest
from unittest.mock import MagicMock, patch, Mock
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class TestRAGSystem:
    """Test RAGSystem orchestration"""

    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        from config import Config
        return Config(
            ANTHROPIC_API_KEY="test_key",
            ANTHROPIC_MODEL="test_model",
            EMBEDDING_MODEL="all-MiniLM-L6-v2",
            CHUNK_SIZE=800,
            CHUNK_OVERLAP=100,
            MAX_RESULTS=5,
            MAX_HISTORY=2,
            CHROMA_PATH="./test_chroma_db"
        )

    @pytest.fixture
    def mock_components(self):
        """Create mocked components"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            yield

    @pytest.fixture
    def rag_system(self, test_config, mock_components):
        """Create RAGSystem instance with mocked components"""
        return RAGSystem(test_config)

    def test_initialization(self, test_config, mock_components):
        """Test RAGSystem initializes all components"""
        system = RAGSystem(test_config)

        assert system.config == test_config
        assert system.document_processor is not None
        assert system.vector_store is not None
        assert system.ai_generator is not None
        assert system.session_manager is not None
        assert system.tool_manager is not None

    def test_query_without_session(self, rag_system):
        """Test query without session ID"""
        # Mock AI generator response
        rag_system.ai_generator.generate_response = MagicMock(return_value="Test response")
        rag_system.tool_manager.get_last_sources = MagicMock(return_value=[])
        rag_system.session_manager.get_conversation_history = MagicMock(return_value=None)

        response, sources = rag_system.query("What is AI?")

        assert response == "Test response"
        assert sources == []

        # Verify AI generator was called
        rag_system.ai_generator.generate_response.assert_called_once()
        call_args = rag_system.ai_generator.generate_response.call_args

        # Verify query was formatted
        assert "What is AI?" in call_args[1]["query"]

    def test_query_with_session(self, rag_system):
        """Test query with session ID and history"""
        # Mock conversation history
        history = "User: Previous question\nAssistant: Previous answer"
        rag_system.session_manager.get_conversation_history = MagicMock(return_value=history)
        rag_system.ai_generator.generate_response = MagicMock(return_value="New response")
        rag_system.tool_manager.get_last_sources = MagicMock(return_value=[])
        rag_system.session_manager.add_exchange = MagicMock()

        response, sources = rag_system.query("Follow up question", session_id="session_1")

        assert response == "New response"

        # Verify history was retrieved
        rag_system.session_manager.get_conversation_history.assert_called_once_with("session_1")

        # Verify history was passed to AI generator
        call_args = rag_system.ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == history

        # Verify exchange was added to history
        rag_system.session_manager.add_exchange.assert_called_once_with(
            "session_1", "Follow up question", "New response"
        )

    def test_query_with_tools(self, rag_system):
        """Test query passes tools to AI generator"""
        rag_system.ai_generator.generate_response = MagicMock(return_value="Response")
        rag_system.tool_manager.get_tool_definitions = MagicMock(return_value=[
            {"name": "search_course_content"}
        ])
        rag_system.tool_manager.get_last_sources = MagicMock(return_value=[])

        response, sources = rag_system.query("Search for AI")

        # Verify tools were passed
        call_args = rag_system.ai_generator.generate_response.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tools"] == [{"name": "search_course_content"}]
        assert "tool_manager" in call_args[1]

    def test_query_source_extraction(self, rag_system):
        """Test that sources are extracted from tool manager"""
        rag_system.ai_generator.generate_response = MagicMock(return_value="Response")
        mock_sources = [
            {"text": "Course 1 - Lesson 1", "link": "http://link1"},
            {"text": "Course 2 - Lesson 2", "link": "http://link2"}
        ]
        rag_system.tool_manager.get_last_sources = MagicMock(return_value=mock_sources)
        rag_system.tool_manager.reset_sources = MagicMock()

        response, sources = rag_system.query("Test query")

        assert sources == mock_sources

        # Verify sources were retrieved and reset
        rag_system.tool_manager.get_last_sources.assert_called_once()
        rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_prompt_formatting(self, rag_system):
        """Test that query is formatted correctly in prompt"""
        rag_system.ai_generator.generate_response = MagicMock(return_value="Response")
        rag_system.tool_manager.get_last_sources = MagicMock(return_value=[])

        rag_system.query("What are neural networks?")

        call_args = rag_system.ai_generator.generate_response.call_args
        query_arg = call_args[1]["query"]

        assert "What are neural networks?" in query_arg
        assert "course materials" in query_arg

    def test_add_course_document_success(self, rag_system, sample_course, sample_course_chunks):
        """Test adding a course document successfully"""
        # Mock document processor
        rag_system.document_processor.process_course_document = MagicMock(
            return_value=(sample_course, sample_course_chunks)
        )
        rag_system.vector_store.add_course_metadata = MagicMock()
        rag_system.vector_store.add_course_content = MagicMock()

        course, num_chunks = rag_system.add_course_document("test.txt")

        assert course == sample_course
        assert num_chunks == len(sample_course_chunks)

        # Verify vector store methods were called
        rag_system.vector_store.add_course_metadata.assert_called_once_with(sample_course)
        rag_system.vector_store.add_course_content.assert_called_once_with(sample_course_chunks)

    def test_add_course_document_failure(self, rag_system):
        """Test adding a course document that fails"""
        rag_system.document_processor.process_course_document = MagicMock(
            side_effect=Exception("File not found")
        )

        course, num_chunks = rag_system.add_course_document("nonexistent.txt")

        assert course is None
        assert num_chunks == 0

    def test_get_course_analytics(self, rag_system):
        """Test getting course analytics"""
        rag_system.vector_store.get_course_count = MagicMock(return_value=3)
        rag_system.vector_store.get_existing_course_titles = MagicMock(
            return_value=["Course 1", "Course 2", "Course 3"]
        )

        analytics = rag_system.get_course_analytics()

        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3
        assert "Course 1" in analytics["course_titles"]

    def test_tool_registration(self, test_config, mock_components):
        """Test that tools are registered on initialization"""
        with patch('rag_system.ToolManager') as mock_tool_manager_class:
            mock_tool_manager = MagicMock()
            mock_tool_manager_class.return_value = mock_tool_manager

            system = RAGSystem(test_config)

            # Verify tools were registered
            assert mock_tool_manager.register_tool.call_count >= 2  # At least search and outline tools

    def test_query_error_propagation(self, rag_system):
        """Test that errors in AI generation are propagated"""
        rag_system.ai_generator.generate_response = MagicMock(
            side_effect=Exception("API error")
        )

        with pytest.raises(Exception, match="API error"):
            rag_system.query("Test query")

    def test_add_course_folder_success(self, rag_system, sample_course, sample_course_chunks):
        """Test adding courses from a folder"""
        with patch('rag_system.os.path.exists', return_value=True), \
             patch('rag_system.os.listdir', return_value=['course1.txt', 'course2.pdf']), \
             patch('rag_system.os.path.isfile', return_value=True):

            rag_system.vector_store.get_existing_course_titles = MagicMock(return_value=[])
            rag_system.document_processor.process_course_document = MagicMock(
                return_value=(sample_course, sample_course_chunks)
            )
            rag_system.vector_store.add_course_metadata = MagicMock()
            rag_system.vector_store.add_course_content = MagicMock()

            total_courses, total_chunks = rag_system.add_course_folder("./docs")

            assert total_courses == 2
            assert total_chunks == len(sample_course_chunks) * 2

    def test_add_course_folder_skip_duplicates(self, rag_system, sample_course, sample_course_chunks):
        """Test that duplicate courses are skipped"""
        with patch('rag_system.os.path.exists', return_value=True), \
             patch('rag_system.os.listdir', return_value=['course1.txt']), \
             patch('rag_system.os.path.isfile', return_value=True):

            # Mock existing course
            rag_system.vector_store.get_existing_course_titles = MagicMock(
                return_value=[sample_course.title]
            )
            rag_system.document_processor.process_course_document = MagicMock(
                return_value=(sample_course, sample_course_chunks)
            )

            total_courses, total_chunks = rag_system.add_course_folder("./docs")

            # Should skip the duplicate
            assert total_courses == 0
            assert total_chunks == 0

    def test_add_course_folder_clear_existing(self, rag_system):
        """Test clearing existing data when adding folder"""
        with patch('rag_system.os.path.exists', return_value=True), \
             patch('rag_system.os.listdir', return_value=[]):

            rag_system.vector_store.clear_all_data = MagicMock()
            rag_system.vector_store.get_existing_course_titles = MagicMock(return_value=[])

            rag_system.add_course_folder("./docs", clear_existing=True)

            # Verify clear was called
            rag_system.vector_store.clear_all_data.assert_called_once()

    def test_query_updates_history_correctly(self, rag_system):
        """Test that query correctly updates conversation history"""
        session_id = "test_session"
        query_text = "What is machine learning?"
        response_text = "Machine learning is a subset of AI."

        rag_system.session_manager.get_conversation_history = MagicMock(return_value=None)
        rag_system.ai_generator.generate_response = MagicMock(return_value=response_text)
        rag_system.tool_manager.get_last_sources = MagicMock(return_value=[])
        rag_system.session_manager.add_exchange = MagicMock()

        rag_system.query(query_text, session_id=session_id)

        # Verify add_exchange was called with correct parameters
        rag_system.session_manager.add_exchange.assert_called_once_with(
            session_id, query_text, response_text
        )
