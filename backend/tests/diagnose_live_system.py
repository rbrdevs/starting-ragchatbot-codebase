"""
Diagnostic script to test the live RAG system and identify issues.
Run this to diagnose why content queries might be failing.
"""

import sys
import os
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config
from vector_store import VectorStore
from search_tools import CourseSearchTool
from ai_generator import AIGenerator
from rag_system import RAGSystem


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_vector_store():
    """Test if vector store has data and can search"""
    print_section("1. Testing Vector Store")

    try:
        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )

        # Check course count
        course_count = store.get_course_count()
        print(f"✓ Total courses in database: {course_count}")

        if course_count == 0:
            print("✗ ERROR: No courses found in database!")
            print("  → This would cause 'query failed' errors")
            return False

        # Get course titles
        titles = store.get_existing_course_titles()
        print(f"✓ Course titles:")
        for title in titles:
            print(f"  - {title}")

        # Try a simple search
        print("\n✓ Testing search functionality...")
        results = store.search(query="introduction to AI", limit=3)

        if results.error:
            print(f"✗ ERROR: Search returned error: {results.error}")
            return False

        if results.is_empty():
            print("✗ WARNING: Search returned no results")
            print("  → This might cause 'No relevant content found' messages")
        else:
            print(f"✓ Search returned {len(results.documents)} results")
            print(f"  First result preview: {results.documents[0][:100]}...")

        return True

    except Exception as e:
        print(f"✗ EXCEPTION in VectorStore: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_course_search_tool():
    """Test if CourseSearchTool works correctly"""
    print_section("2. Testing CourseSearchTool")

    try:
        store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        tool = CourseSearchTool(store)

        print("✓ Tool definition:")
        definition = tool.get_tool_definition()
        print(f"  Name: {definition['name']}")
        print(f"  Required params: {definition['input_schema']['required']}")

        print("\n✓ Testing tool execution...")
        result = tool.execute(query="what is machine learning")

        if "Search error" in result or "query failed" in result.lower():
            print(f"✗ ERROR: Tool returned error message:")
            print(f"  {result}")
            return False

        if "No relevant content found" in result:
            print("✗ WARNING: No content found for query")
            print(f"  Result: {result}")
        else:
            print(f"✓ Tool executed successfully")
            print(f"  Result preview: {result[:200]}...")
            print(f"  Sources tracked: {len(tool.last_sources)}")

        return True

    except Exception as e:
        print(f"✗ EXCEPTION in CourseSearchTool: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ai_generator():
    """Test if AI Generator is configured correctly"""
    print_section("3. Testing AI Generator Configuration")

    try:
        # Check API key
        api_key = config.ANTHROPIC_API_KEY
        if not api_key or api_key == "":
            print("✗ ERROR: ANTHROPIC_API_KEY is not set!")
            print("  → This would cause API errors")
            return False

        print(f"✓ API Key is set: {api_key[:10]}...")
        print(f"✓ Model: {config.ANTHROPIC_MODEL}")

        # Note: We won't actually call the API to avoid charges
        print("✓ AI Generator configuration looks correct")
        print("  (Skipping actual API call to avoid charges)")

        return True

    except Exception as e:
        print(f"✗ EXCEPTION in AI Generator check: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_rag_system():
    """Test the complete RAG system"""
    print_section("4. Testing RAG System Integration")

    try:
        rag = RAGSystem(config)

        print("✓ RAG system initialized")

        # Check analytics
        analytics = rag.get_course_analytics()
        print(f"✓ Course analytics:")
        print(f"  Total courses: {analytics['total_courses']}")

        if analytics["total_courses"] == 0:
            print("✗ ERROR: No courses loaded!")
            return False

        # Test tool registration
        tool_defs = rag.tool_manager.get_tool_definitions()
        print(f"✓ Registered tools: {len(tool_defs)}")
        for tool_def in tool_defs:
            print(f"  - {tool_def['name']}")

        print("\n✓ RAG system appears functional")
        return True

    except Exception as e:
        print(f"✗ EXCEPTION in RAG System: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_actual_query():
    """Test an actual query to see what happens"""
    print_section("5. Testing Actual Query (No API Call)")

    try:
        # We'll trace through the query without actually calling the API
        from unittest.mock import MagicMock

        rag = RAGSystem(config)

        # Mock the AI generator to avoid API calls
        original_generate = rag.ai_generator.generate_response
        rag.ai_generator.generate_response = MagicMock(return_value="Mocked response")

        print("✓ Testing query: 'What is deep learning?'")

        try:
            response, sources = rag.query("What is deep learning?")
            print(f"✓ Query executed successfully")
            print(f"  Response: {response}")
            print(f"  Sources: {sources}")
        except Exception as query_error:
            print(f"✗ ERROR during query execution:")
            print(f"  {query_error}")
            import traceback

            traceback.print_exc()
            return False

        return True

    except Exception as e:
        print(f"✗ EXCEPTION in query test: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests"""
    print("\n" + "=" * 70)
    print(" RAG SYSTEM DIAGNOSTIC TOOL")
    print("=" * 70)

    results = {
        "Vector Store": test_vector_store(),
        "Course Search Tool": test_course_search_tool(),
        "AI Generator Config": test_ai_generator(),
        "RAG System": test_rag_system(),
        "Actual Query": test_actual_query(),
    }

    print_section("DIAGNOSTIC SUMMARY")
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:25} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All diagnostic tests passed!")
        print("  If you're still getting 'query failed' errors, the issue might be:")
        print("  1. API key permissions")
        print("  2. Network/firewall blocking Anthropic API")
        print("  3. Rate limiting")
    else:
        print("\n✗ Some tests failed - see details above")
        print("  The failing components are likely causing the 'query failed' errors")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
