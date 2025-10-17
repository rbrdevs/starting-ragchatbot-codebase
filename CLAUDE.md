# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **tool-based RAG (Retrieval-Augmented Generation) chatbot** that answers questions about course materials. The system uses ChromaDB for vector storage, Anthropic's Claude API for AI generation with tool calling, and provides a web interface for interaction.

**Key Architecture Pattern**: Claude AI autonomously decides when to search course materials using tools, rather than searching on every query. This is a two-phase generation pattern: (1) Claude requests tool execution, (2) system executes search and returns results, (3) Claude synthesizes final answer.

## Development Commands

### Setup
```bash
# Install dependencies
uv sync

# Create .env file (required)
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start with hot-reload
cd backend
uv run uvicorn app:app --reload --port 8000

# Custom port
cd backend
uv run uvicorn app:app --reload --port 8080
```

### Accessing the Application
- Web UI: http://localhost:8000
- API docs (Swagger): http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

### Development
```bash
# Run Python scripts directly with uv
uv run python backend/script.py

# Interactive Python shell with dependencies
uv run python
```

## Architecture Overview

### RAG Flow (Query → Response)

The system uses a **12-phase pipeline** from user input to rendered response:

1. **Frontend** (`script.js`) → POST `/api/query` with `{query, session_id}`
2. **FastAPI** (`app.py`) → Validates request, routes to RAG system
3. **RAG System** (`rag_system.py`) → Orchestrates components, builds prompt
4. **Session Manager** (`session_manager.py`) → Retrieves conversation history (max 2 exchanges)
5. **AI Generator** (`ai_generator.py`) → First Claude API call with tool definitions
6. **Claude API** → Decides whether to use `search_course_content` tool
7. **Tool Manager** (`search_tools.py`) → Routes tool execution
8. **Vector Store** (`vector_store.py`) → Semantic search in ChromaDB
9. **Search Tool** → Formats results with `[Course - Lesson N]` headers, tracks sources
10. **AI Generator** → Second Claude API call with tool results (no tools this time)
11. **RAG System** → Extracts sources, updates session history
12. **Frontend** → Renders markdown response with collapsible sources

### Critical Design Patterns

**Two-Phase AI Interaction**:
- First API call: Claude receives tools, decides to search or answer directly
- If tool used: Execute search, return results as `tool_result` message
- Second API call: Claude synthesizes answer from tool results (no tools provided to prevent loops)

**Dual-Collection Vector Storage**:
- `course_catalog`: Course metadata for semantic course name matching
- `course_content`: Text chunks with embeddings for content search
- Course name resolution happens first via semantic search on catalog, then filters content

**Tool-Based RAG vs Traditional RAG**:
- Traditional: Always retrieves context for every query
- This system: Claude decides when retrieval is needed (e.g., "hello" doesn't trigger search)

### Component Responsibilities

**RAGSystem** (`rag_system.py`) - Main orchestrator:
- Coordinates all components (document processor, vector store, AI generator, session manager)
- Manages tool definitions and source tracking
- Single entry point: `query(query, session_id) -> (answer, sources)`

**AIGenerator** (`ai_generator.py`) - Claude API wrapper:
- Handles two-phase tool execution workflow
- System prompt defines Claude's behavior and tool usage rules
- Temperature=0 for deterministic responses
- Max tokens=800 (configurable in `config.py`)

**VectorStore** (`vector_store.py`) - ChromaDB interface:
- `search()`: Unified search with course name resolution and metadata filtering
- `_resolve_course_name()`: Semantic matching for fuzzy course names
- `_build_filter()`: Constructs ChromaDB filters for course/lesson constraints
- Uses `all-MiniLM-L6-v2` embedding model (384 dimensions)

**DocumentProcessor** (`document_processor.py`) - Text chunking:
- Expected format: `Course Title:`, `Course Link:`, `Course Instructor:`, then `Lesson N:` markers
- Chunks: 800 chars with 100-char overlap (sentence-aware splitting)
- First chunk of each lesson gets context prefix: `"Lesson N content: [text]"`
- Regex pattern for lessons: `^Lesson\s+(\d+):\s*(.+)$`

**SessionManager** (`session_manager.py`) - Conversation state:
- Stores Message objects: `{role: "user"|"assistant", content: str}`
- Default max history: 2 exchanges (4 messages total)
- History formatted as: `"User: question\nAssistant: answer\n..."`
- Auto-prunes old messages when limit exceeded

**ToolManager** (`search_tools.py`) - Tool registry:
- Registers tools that implement `Tool` interface (ABC with `get_tool_definition()` and `execute()`)
- Routes tool calls by name
- Tracks sources from last search via `last_sources` attribute on CourseSearchTool
- Sources must be manually reset after retrieval

### Data Models

**Course** → **Lessons** → **CourseChunks**:
- `Course`: Contains title (unique ID), instructor, course_link, lessons list
- `Lesson`: lesson_number, title, lesson_link
- `CourseChunk`: content (with context prefix), course_title, lesson_number, chunk_index

**ChromaDB Storage**:
- IDs for catalog: `course.title`
- IDs for content: `f"{course_title.replace(' ', '_')}_{chunk_index}"`
- Metadata includes: course_title, lesson_number, chunk_index

### Configuration (`config.py`)

Important defaults:
- Model: `claude-sonnet-4-5-20250514`
- Chunk size: 800 chars with 100 overlap
- Max search results: 5
- Max conversation history: 2 exchanges
- Embedding model: `all-MiniLM-L6-v2`
- ChromaDB path: `./chroma_db`

## Document Processing

### Expected Document Format

Course files must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [lesson title]
Lesson Link: [url]
[lesson content...]

Lesson 1: [lesson title]
Lesson Link: [url]
[lesson content...]
```

### Adding New Documents

Documents in `/docs` are loaded automatically on startup:
- Place `.txt`, `.pdf`, or `.docx` files in `/docs` folder
- Restart server to process new documents
- Duplicate courses (by title) are skipped
- Processing happens in `app.py:startup_event()`

To force rebuild of vector database:
```python
# In rag_system.py
rag_system.add_course_folder("../docs", clear_existing=True)
```

## Tool System

### Adding New Tools

1. Create class implementing `Tool` ABC from `search_tools.py`:
```python
class MyTool(Tool):
    def get_tool_definition(self) -> Dict[str, Any]:
        return {
            "name": "my_tool",
            "description": "What this tool does",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "..."}
                },
                "required": ["param"]
            }
        }

    def execute(self, **kwargs) -> str:
        # Tool implementation
        return "result string"
```

2. Register in `RAGSystem.__init__()`:
```python
my_tool = MyTool()
self.tool_manager.register_tool(my_tool)
```

### Tool Execution Flow

Claude's tool use block looks like:
```json
{
  "type": "tool_use",
  "id": "toolu_abc123",
  "name": "search_course_content",
  "input": {"query": "...", "course_name": null, "lesson_number": null}
}
```

System responds with:
```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_abc123",
  "content": "[Course - Lesson N]\nformatted results..."
}
```

## Frontend Integration

### API Endpoints

**POST /api/query**:
- Request: `{query: str, session_id?: str}`
- Response: `{answer: str, sources: List[str], session_id: str}`
- Creates session if `session_id` is null

**GET /api/courses**:
- Response: `{total_courses: int, course_titles: List[str]}`

### Frontend Architecture

- `script.js`: Handles API calls, markdown rendering (marked.js), session management
- `index.html`: Single-page app with chat interface and sidebar
- `style.css`: Dark theme with collapsible sources

Key frontend features:
- Loading animation while waiting for response
- Markdown rendering for assistant messages (user messages are escaped HTML)
- Collapsible `<details>` for sources
- Session persistence via `currentSessionId` global variable

## Modifying AI Behavior

### System Prompt

Located in `ai_generator.py:SYSTEM_PROMPT`. Key instructions:
- "Use the search tool **only** for questions about specific course content"
- "**One search per query maximum**"
- "**No meta-commentary**" - no "based on the search results" phrases
- "Brief, Concise and focused"

### Changing Search Strategy

To modify when Claude searches:
1. Edit system prompt in `ai_generator.py`
2. Add/remove examples of when to search
3. Adjust `tool_choice` parameter (currently `"auto"`, could be `"required"` or specific tool)

### Adjusting Context Window

- Conversation history: Change `MAX_HISTORY` in `config.py` (default: 2 exchanges)
- Search results: Change `MAX_RESULTS` in `config.py` (default: 5 chunks)
- Max tokens: Change `max_tokens` in `ai_generator.py` (default: 800)

## Common Development Scenarios

### Changing the AI Model

Edit `config.py`:
```python
ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"  # or other model
```

### Adjusting Chunk Size

Larger chunks = more context per result, fewer total chunks:
```python
# config.py
CHUNK_SIZE: int = 1200  # default: 800
CHUNK_OVERLAP: int = 150  # default: 100
```

Must rebuild vector DB after changing chunk settings.

### Adding Metadata Filters

To filter by additional criteria (e.g., instructor, course type):
1. Add field to `CourseChunk` model in `models.py`
2. Include in metadata when adding to vector store in `vector_store.py:add_course_content()`
3. Extend `_build_filter()` in `vector_store.py` to support new filter
4. Add parameter to tool definition in `search_tools.py`

### Debugging Tool Calls

Enable detailed logging by checking `response.content` in `ai_generator.py`:
```python
print(f"Claude's response: {response.content}")
print(f"Stop reason: {response.stop_reason}")
```

View full conversation in browser at `/api/query` endpoint logs or in FastAPI's `/docs` interface using "Try it out".

## File Organization

```
backend/
├── app.py                 # FastAPI app, startup, endpoints
├── config.py              # Configuration dataclass
├── models.py              # Pydantic models (Course, Lesson, CourseChunk)
├── rag_system.py          # Main orchestrator
├── ai_generator.py        # Claude API wrapper
├── vector_store.py        # ChromaDB interface
├── document_processor.py  # Text parsing and chunking
├── search_tools.py        # Tool definitions and manager
└── session_manager.py     # Conversation state

frontend/
├── index.html            # Single-page app
├── script.js             # API client, UI updates
└── style.css             # Dark theme styles

docs/                     # Course material files (auto-loaded)
chroma_db/               # Vector database (auto-created)
.env                     # ANTHROPIC_API_KEY (required, not in git)
```

## Windows-Specific Notes

- Use Git Bash for running `./run.sh`
- PowerShell users: Run `cd backend; uv run uvicorn app:app --reload --port 8000` directly
- ChromaDB requires SQLite3 - if errors occur, install: `pip install pysqlite3-binary`

## Prerequisites

- Python 3.13+ (uses modern type hints and dataclasses)
- uv package manager (replaces pip/poetry)
- Anthropic API key with Claude access
- ~500MB disk space for embeddings model on first run
