import pytest
from unittest.mock import MagicMock, patch
from vector_store import VectorStore, SearchResults


class TestSearchResults:
    """Test SearchResults dataclass"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2', 'doc3']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}, {'key': 'value3'}]],
            'distances': [[0.1, 0.2, 0.3]]
        }
        results = SearchResults.from_chroma(chroma_results)
        assert len(results.documents) == 3
        assert len(results.metadata) == 3
        assert len(results.distances) == 3
        assert results.error is None

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        results = SearchResults.from_chroma(chroma_results)
        assert results.is_empty()
        assert results.error is None

    def test_empty_with_error(self):
        """Test creating empty results with error message"""
        results = SearchResults.empty("Test error message")
        assert results.is_empty()
        assert results.error == "Test error message"

    def test_is_empty(self):
        """Test is_empty method"""
        empty = SearchResults(documents=[], metadata=[], distances=[])
        assert empty.is_empty()

        non_empty = SearchResults(documents=['doc'], metadata=[{}], distances=[0.1])
        assert not non_empty.is_empty()


class TestVectorStore:
    """Test VectorStore functionality with mocked ChromaDB"""

    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client"""
        with patch('vector_store.chromadb.PersistentClient') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Mock collections
            mock_catalog = MagicMock()
            mock_content = MagicMock()
            mock_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]

            yield mock_instance, mock_catalog, mock_content

    def test_vector_store_initialization(self, mock_chroma_client):
        """Test VectorStore initializes correctly"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5)

        assert store.max_results == 5
        assert mock_instance.get_or_create_collection.call_count == 2

    def test_resolve_course_name_found(self, mock_chroma_client):
        """Test course name resolution when course is found"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client

        # Mock catalog query response
        mock_catalog.query.return_value = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course on AI'}]]
        }

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")
        result = store._resolve_course_name("AI course")

        assert result == 'Test Course on AI'
        mock_catalog.query.assert_called_once()

    def test_resolve_course_name_not_found(self, mock_chroma_client):
        """Test course name resolution when course is not found"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client

        # Mock empty catalog query response
        mock_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")
        result = store._resolve_course_name("nonexistent course")

        assert result is None

    def test_build_filter_no_params(self, mock_chroma_client):
        """Test filter building with no parameters"""
        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")
        filter_dict = store._build_filter(None, None)

        assert filter_dict is None

    def test_build_filter_course_only(self, mock_chroma_client):
        """Test filter building with only course title"""
        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")
        filter_dict = store._build_filter("Test Course", None)

        assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, mock_chroma_client):
        """Test filter building with only lesson number"""
        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")
        filter_dict = store._build_filter(None, 1)

        assert filter_dict == {"lesson_number": 1}

    def test_build_filter_both_params(self, mock_chroma_client):
        """Test filter building with both course and lesson"""
        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")
        filter_dict = store._build_filter("Test Course", 1)

        assert filter_dict == {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 1}
            ]
        }

    def test_search_without_filters(self, mock_chroma_client):
        """Test search without course or lesson filters"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client

        # Mock content query response
        mock_content.query.return_value = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course_title': 'Test', 'lesson_number': 0}, {'course_title': 'Test', 'lesson_number': 1}]],
            'distances': [[0.1, 0.2]]
        }

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5)
        results = store.search(query="neural networks")

        assert len(results.documents) == 2
        assert results.error is None
        mock_content.query.assert_called_once_with(
            query_texts=["neural networks"],
            n_results=5,
            where=None
        )

    def test_search_with_course_filter(self, mock_chroma_client):
        """Test search with course name filter"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client

        # Mock catalog query for course resolution
        mock_catalog.query.return_value = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course on AI'}]]
        }

        # Mock content query response
        mock_content.query.return_value = {
            'documents': [['doc1']],
            'metadatas': [[{'course_title': 'Test Course on AI', 'lesson_number': 0}]],
            'distances': [[0.1]]
        }

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5)
        results = store.search(query="neural networks", course_name="AI course")

        assert len(results.documents) == 1
        assert results.error is None
        # Verify the filter was applied
        call_args = mock_content.query.call_args
        assert call_args[1]['where'] == {"course_title": "Test Course on AI"}

    def test_search_with_invalid_course(self, mock_chroma_client):
        """Test search with invalid course name returns error"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client

        # Mock catalog query returns no results
        mock_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")
        results = store.search(query="test", course_name="nonexistent")

        assert results.error == "No course found matching 'nonexistent'"
        assert results.is_empty()

    def test_search_with_lesson_filter(self, mock_chroma_client):
        """Test search with lesson number filter"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client

        # Mock content query response
        mock_content.query.return_value = {
            'documents': [['doc1']],
            'metadatas': [[{'course_title': 'Test', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5)
        results = store.search(query="test", lesson_number=1)

        assert len(results.documents) == 1
        call_args = mock_content.query.call_args
        assert call_args[1]['where'] == {"lesson_number": 1}

    def test_search_with_both_filters(self, mock_chroma_client):
        """Test search with both course and lesson filters"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client

        # Mock catalog query
        mock_catalog.query.return_value = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course on AI'}]]
        }

        # Mock content query response
        mock_content.query.return_value = {
            'documents': [['doc1']],
            'metadatas': [[{'course_title': 'Test Course on AI', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5)
        results = store.search(query="test", course_name="AI", lesson_number=1)

        assert len(results.documents) == 1
        call_args = mock_content.query.call_args
        expected_filter = {
            "$and": [
                {"course_title": "Test Course on AI"},
                {"lesson_number": 1}
            ]
        }
        assert call_args[1]['where'] == expected_filter

    def test_search_exception_handling(self, mock_chroma_client):
        """Test search handles exceptions gracefully"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client

        # Mock content query to raise exception
        mock_content.query.side_effect = Exception("Database error")

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")
        results = store.search(query="test")

        assert results.error == "Search error: Database error"
        assert results.is_empty()

    def test_search_with_custom_limit(self, mock_chroma_client):
        """Test search with custom result limit"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client

        # Mock content query response
        mock_content.query.return_value = {
            'documents': [['doc1', 'doc2', 'doc3']],
            'metadatas': [[{}, {}, {}]],
            'distances': [[0.1, 0.2, 0.3]]
        }

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2", max_results=5)
        results = store.search(query="test", limit=3)

        call_args = mock_content.query.call_args
        assert call_args[1]['n_results'] == 3

    def test_get_existing_course_titles(self, mock_chroma_client):
        """Test getting all existing course titles"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client

        # Mock catalog get response
        mock_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")
        titles = store.get_existing_course_titles()

        assert len(titles) == 3
        assert 'Course 1' in titles

    def test_get_lesson_link(self, mock_chroma_client):
        """Test getting lesson link for a course and lesson"""
        mock_instance, mock_catalog, mock_content = mock_chroma_client

        # Mock catalog get response
        import json
        lessons_data = [
            {"lesson_number": 0, "lesson_title": "Intro", "lesson_link": "http://lesson0"},
            {"lesson_number": 1, "lesson_title": "Basics", "lesson_link": "http://lesson1"}
        ]
        mock_catalog.get.return_value = {
            'metadatas': [{
                'lessons_json': json.dumps(lessons_data)
            }]
        }

        store = VectorStore(chroma_path="./test_db", embedding_model="all-MiniLM-L6-v2")
        link = store.get_lesson_link("Test Course", 1)

        assert link == "http://lesson1"
