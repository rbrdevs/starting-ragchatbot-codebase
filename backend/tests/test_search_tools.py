import pytest
from unittest.mock import MagicMock, Mock
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test CourseSearchTool functionality"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore"""
        return MagicMock()

    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create a CourseSearchTool instance with mock store"""
        return CourseSearchTool(mock_vector_store)

    def test_get_tool_definition(self, search_tool):
        """Test that tool definition is correctly structured"""
        definition = search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]

    def test_execute_with_valid_query_no_filters(self, search_tool, mock_vector_store):
        """Test execute with valid query and no filters"""
        # Mock search results
        mock_results = SearchResults(
            documents=["This is test content about neural networks."],
            metadata=[{"course_title": "AI Basics", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "http://lesson1"

        result = search_tool.execute(query="neural networks")

        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="neural networks", course_name=None, lesson_number=None
        )

        # Verify result format
        assert "[AI Basics - Lesson 1]" in result
        assert "This is test content" in result

    def test_execute_with_course_filter(self, search_tool, mock_vector_store):
        """Test execute with course name filter"""
        mock_results = SearchResults(
            documents=["Course specific content."],
            metadata=[{"course_title": "Deep Learning", "lesson_number": 2}],
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "http://lesson2"

        result = search_tool.execute(query="test", course_name="Deep Learning")

        mock_vector_store.search.assert_called_once_with(
            query="test", course_name="Deep Learning", lesson_number=None
        )
        assert "[Deep Learning - Lesson 2]" in result

    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store):
        """Test execute with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson specific content."],
            metadata=[{"course_title": "ML Course", "lesson_number": 3}],
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "http://lesson3"

        result = search_tool.execute(query="test", lesson_number=3)

        mock_vector_store.search.assert_called_once_with(
            query="test", course_name=None, lesson_number=3
        )
        assert "[ML Course - Lesson 3]" in result

    def test_execute_with_both_filters(self, search_tool, mock_vector_store):
        """Test execute with both course and lesson filters"""
        mock_results = SearchResults(
            documents=["Filtered content."],
            metadata=[{"course_title": "AI Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "http://lesson1"

        result = search_tool.execute(
            query="test", course_name="AI Course", lesson_number=1
        )

        mock_vector_store.search.assert_called_once_with(
            query="test", course_name="AI Course", lesson_number=1
        )

    def test_execute_with_error_result(self, search_tool, mock_vector_store):
        """Test execute when search returns an error"""
        mock_results = SearchResults.empty("Search error: Database connection failed")
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(query="test")

        assert result == "Search error: Database connection failed"

    def test_execute_with_empty_results_no_filters(
        self, search_tool, mock_vector_store
    ):
        """Test execute with empty results and no filters"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(query="nonexistent topic")

        assert result == "No relevant content found."

    def test_execute_with_empty_results_with_course_filter(
        self, search_tool, mock_vector_store
    ):
        """Test execute with empty results and course filter"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(query="test", course_name="AI Course")

        assert result == "No relevant content found in course 'AI Course'."

    def test_execute_with_empty_results_with_lesson_filter(
        self, search_tool, mock_vector_store
    ):
        """Test execute with empty results and lesson filter"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(query="test", lesson_number=5)

        assert result == "No relevant content found in lesson 5."

    def test_execute_with_empty_results_both_filters(
        self, search_tool, mock_vector_store
    ):
        """Test execute with empty results and both filters"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(query="test", course_name="AI", lesson_number=1)

        assert "No relevant content found in course 'AI' in lesson 1." == result

    def test_format_results_multiple_documents(self, search_tool, mock_vector_store):
        """Test formatting of multiple search results"""
        mock_results = SearchResults(
            documents=["Content 1", "Content 2", "Content 3"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course A", "lesson_number": 2},
                {"course_title": "Course B", "lesson_number": 1},
            ],
            distances=[0.1, 0.2, 0.3],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "http://lesson"

        result = search_tool.execute(query="test")

        assert "[Course A - Lesson 1]" in result
        assert "[Course A - Lesson 2]" in result
        assert "[Course B - Lesson 1]" in result
        assert "Content 1" in result
        assert "Content 2" in result
        assert "Content 3" in result

    def test_format_results_without_lesson_number(self, search_tool, mock_vector_store):
        """Test formatting when lesson_number is None"""
        mock_results = SearchResults(
            documents=["General course content"],
            metadata=[{"course_title": "General Course", "lesson_number": None}],
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None

        result = search_tool.execute(query="test")

        assert "[General Course]" in result
        assert "General course content" in result

    def test_source_tracking(self, search_tool, mock_vector_store):
        """Test that sources are tracked correctly"""
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "http://lesson1"

        result = search_tool.execute(query="test")

        # Verify sources were tracked
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]["text"] == "Test Course - Lesson 1"
        assert search_tool.last_sources[0]["link"] == "http://lesson1"

    def test_source_tracking_multiple_results(self, search_tool, mock_vector_store):
        """Test source tracking with multiple results"""
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.side_effect = ["http://a1", "http://b2"]

        result = search_tool.execute(query="test")

        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert search_tool.last_sources[0]["link"] == "http://a1"
        assert search_tool.last_sources[1]["text"] == "Course B - Lesson 2"
        assert search_tool.last_sources[1]["link"] == "http://b2"


class TestCourseOutlineTool:
    """Test CourseOutlineTool functionality"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore"""
        return MagicMock()

    @pytest.fixture
    def outline_tool(self, mock_vector_store):
        """Create a CourseOutlineTool instance with mock store"""
        return CourseOutlineTool(mock_vector_store)

    def test_get_tool_definition(self, outline_tool):
        """Test that tool definition is correctly structured"""
        definition = outline_tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert "course_name" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_name"]

    def test_execute_with_valid_course(self, outline_tool, mock_vector_store):
        """Test execute with valid course name"""
        import json

        # Mock course name resolution
        mock_vector_store._resolve_course_name.return_value = "AI Fundamentals"

        # Mock course catalog data
        lessons_data = [
            {"lesson_number": 0, "lesson_title": "Introduction"},
            {"lesson_number": 1, "lesson_title": "Basics"},
        ]
        mock_vector_store.course_catalog.get.return_value = {
            "metadatas": [
                {
                    "title": "AI Fundamentals",
                    "course_link": "http://course",
                    "lessons_json": json.dumps(lessons_data),
                }
            ]
        }

        result = outline_tool.execute(course_name="AI")

        assert "Course: AI Fundamentals" in result
        assert "Link: http://course" in result
        assert "Lessons:" in result
        assert "Lesson 0: Introduction" in result
        assert "Lesson 1: Basics" in result

    def test_execute_with_nonexistent_course(self, outline_tool, mock_vector_store):
        """Test execute with course that doesn't exist"""
        mock_vector_store._resolve_course_name.return_value = None

        result = outline_tool.execute(course_name="Nonexistent Course")

        assert result == "No course found matching 'Nonexistent Course'"

    def test_execute_with_metadata_error(self, outline_tool, mock_vector_store):
        """Test execute when metadata retrieval fails"""
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.course_catalog.get.return_value = None

        result = outline_tool.execute(course_name="Test")

        assert "No metadata found" in result


class TestToolManager:
    """Test ToolManager functionality"""

    def test_register_tool(self):
        """Test registering a tool"""
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "Test tool",
        }

        manager.register_tool(mock_tool)

        assert "test_tool" in manager.tools

    def test_register_tool_without_name(self):
        """Test registering a tool without name raises error"""
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"description": "Test"}

        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)

    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        manager = ToolManager()
        mock_tool1 = MagicMock()
        mock_tool1.get_tool_definition.return_value = {"name": "tool1"}
        mock_tool2 = MagicMock()
        mock_tool2.get_tool_definition.return_value = {"name": "tool2"}

        manager.register_tool(mock_tool1)
        manager.register_tool(mock_tool2)

        definitions = manager.get_tool_definitions()
        assert len(definitions) == 2

    def test_execute_tool(self):
        """Test executing a registered tool"""
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"name": "test_tool"}
        mock_tool.execute.return_value = "Tool result"

        manager.register_tool(mock_tool)
        result = manager.execute_tool("test_tool", param1="value1")

        assert result == "Tool result"
        mock_tool.execute.assert_called_once_with(param1="value1")

    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool")

        assert result == "Tool 'nonexistent_tool' not found"

    def test_get_last_sources(self):
        """Test getting sources from tools"""
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"name": "search_tool"}
        mock_tool.last_sources = [{"text": "Source 1", "link": "http://link"}]

        manager.register_tool(mock_tool)
        sources = manager.get_last_sources()

        assert len(sources) == 1
        assert sources[0]["text"] == "Source 1"

    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        manager = ToolManager()
        mock_tool1 = MagicMock()
        mock_tool1.get_tool_definition.return_value = {"name": "tool1"}
        mock_tool1.last_sources = [{"text": "Source"}]

        manager.register_tool(mock_tool1)
        manager.reset_sources()

        assert mock_tool1.last_sources == []
