# RAG Chatbot Query Flow Diagram

## Complete End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION (Browser)                       │
│                                                                           │
│  User: "What is prompt caching?"                                         │
│  ┌─────────────────────────┐                                            │
│  │  Chat Input + Send Btn  │                                            │
│  └───────────┬─────────────┘                                            │
└──────────────┼──────────────────────────────────────────────────────────┘
               │
               │ POST /api/query
               │ {query: "What is prompt caching?", session_id: "session_1"}
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: API ENDPOINT (app.py)                        │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ @app.post("/api/query")                                      │        │
│  │ async def query_documents(request: QueryRequest)             │        │
│  │   1. Get/Create session_id                                   │        │
│  │   2. Call rag_system.query(query, session_id)               │        │
│  └────────────────────────┬────────────────────────────────────┘        │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  PHASE 2: RAG ORCHESTRATION (rag_system.py)              │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  def query(query, session_id):                            │           │
│  │    1. Build prompt                                        │           │
│  │    2. Get conversation history ──┐                        │           │
│  │    3. Get tool definitions ──┐   │                        │           │
│  │    4. Call AI generator      │   │                        │           │
│  └──────────────────────────────┼───┼────────────────────────┘           │
└─────────────────────────────────┼───┼──────────────────────────────────┘
                                  │   │
                ┌─────────────────┘   └──────────────────┐
                │                                         │
                ▼                                         ▼
   ┌───────────────────────────┐           ┌──────────────────────────────┐
   │  TOOL MANAGER             │           │  SESSION MANAGER             │
   │  (search_tools.py)        │           │  (session_manager.py)        │
   │                           │           │                              │
   │  get_tool_definitions()   │           │  get_conversation_history()  │
   │  Returns:                 │           │  Returns:                    │
   │  [{                       │           │  "User: prev question\n      │
   │    name: "search_...",    │           │   Assistant: prev answer"    │
   │    description: "...",    │           │                              │
   │    input_schema: {...}    │           │  Max history: 2 exchanges    │
   │  }]                       │           │  (4 messages total)          │
   └───────────────────────────┘           └──────────────────────────────┘
                │                                         │
                └─────────────────┬───────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                PHASE 3: AI GENERATION (ai_generator.py)                  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  def generate_response(query, history, tools, tool_mgr)  │           │
│  │                                                            │           │
│  │  1. Build system prompt:                                  │           │
│  │     "You are an AI assistant... [instructions]            │           │
│  │      Previous conversation: [history]"                    │           │
│  │                                                            │           │
│  │  2. Prepare API call parameters:                          │           │
│  │     - model: claude-sonnet-4-5                            │           │
│  │     - temperature: 0                                      │           │
│  │     - max_tokens: 800                                     │           │
│  │     - messages: [{"role":"user","content":query}]         │           │
│  │     - system: [system prompt with history]                │           │
│  │     - tools: [tool definitions]                           │           │
│  │     - tool_choice: {"type": "auto"}                       │           │
│  └────────────────────────┬─────────────────────────────────┘           │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            │ API Call #1
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ANTHROPIC CLAUDE API (First Call)                     │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  Claude analyzes the query:                               │           │
│  │  "What is prompt caching?"                                │           │
│  │                                                            │           │
│  │  Decision Tree:                                            │           │
│  │  ┌─────────────────────────────────────────────┐          │           │
│  │  │ Is this about specific course content?      │          │           │
│  │  │   YES → Use search_course_content tool      │          │           │
│  │  │   NO  → Answer with general knowledge       │          │           │
│  │  └─────────────────────────────────────────────┘          │           │
│  │                                                            │           │
│  │  Response:                                                 │           │
│  │  {                                                         │           │
│  │    stop_reason: "tool_use",                               │           │
│  │    content: [                                              │           │
│  │      {type: "text", text: "Let me search..."},            │           │
│  │      {                                                     │           │
│  │        type: "tool_use",                                   │           │
│  │        id: "toolu_abc123",                                 │           │
│  │        name: "search_course_content",                      │           │
│  │        input: {                                            │           │
│  │          query: "prompt caching cost latency",             │           │
│  │          course_name: null,                                │           │
│  │          lesson_number: null                               │           │
│  │        }                                                    │           │
│  │      }                                                      │           │
│  │    ]                                                        │           │
│  │  }                                                          │           │
│  └────────────────────────┬───────────────────────────────────┘           │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                PHASE 4: TOOL EXECUTION (ai_generator.py)                 │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  if response.stop_reason == "tool_use":                   │           │
│  │    return _handle_tool_execution(...)                     │           │
│  │                                                            │           │
│  │  1. Extract tool call details:                            │           │
│  │     - tool_name: "search_course_content"                  │           │
│  │     - tool_id: "toolu_abc123"                             │           │
│  │     - input: {query, course_name, lesson_number}          │           │
│  │                                                            │           │
│  │  2. Execute tool via ToolManager:                         │           │
│  │     tool_manager.execute_tool(name, **input)              │           │
│  └────────────────────────┬───────────────────────────────────┘           │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PHASE 5: SEARCH TOOL ROUTING (search_tools.py)              │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  ToolManager.execute_tool("search_course_content", ...)  │           │
│  │    ↓                                                      │           │
│  │  CourseSearchTool.execute(                                │           │
│  │    query="prompt caching cost latency",                   │           │
│  │    course_name=None,                                      │           │
│  │    lesson_number=None                                     │           │
│  │  )                                                         │           │
│  └────────────────────────┬───────────────────────────────────┘           │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PHASE 6: VECTOR SEARCH (vector_store.py)                    │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  VectorStore.search(query, course_name, lesson_number)       │       │
│  │                                                                │       │
│  │  Step 1: Resolve course name (if provided)                    │       │
│  │  ┌────────────────────────────────────────┐                  │       │
│  │  │ If course_name provided:               │                  │       │
│  │  │   Search course_catalog collection     │                  │       │
│  │  │   with semantic similarity             │                  │       │
│  │  │   Returns: exact course title          │                  │       │
│  │  └────────────────────────────────────────┘                  │       │
│  │                                                                │       │
│  │  Step 2: Build metadata filter                                │       │
│  │  ┌────────────────────────────────────────┐                  │       │
│  │  │ filter_dict = {                        │                  │       │
│  │  │   "course_title": "...",  # if course  │                  │       │
│  │  │   "lesson_number": 5      # if lesson  │                  │       │
│  │  │ }                                       │                  │       │
│  │  └────────────────────────────────────────┘                  │       │
│  │                                                                │       │
│  │  Step 3: Semantic search on course_content                    │       │
│  │  ┌─────────────────────────────────────────────────────┐     │       │
│  │  │  query: "prompt caching cost latency"               │     │       │
│  │  │         ↓ [Embedding via SentenceTransformer]      │     │       │
│  │  │  embedding: [0.23, -0.45, 0.67, ..., 0.12]         │     │       │
│  │  │         ↓                                            │     │       │
│  │  │  ┌──────────────────────────────────────┐          │     │       │
│  │  │  │     CHROMADB SEARCH                  │          │     │       │
│  │  │  │                                       │          │     │       │
│  │  │  │  Collection: course_content          │          │     │       │
│  │  │  │  Documents: 400+ chunks               │          │     │       │
│  │  │  │                                       │          │     │       │
│  │  │  │  1. Compare query embedding with     │          │     │       │
│  │  │  │     all chunk embeddings (cosine)    │          │     │       │
│  │  │  │  2. Apply metadata filters           │          │     │       │
│  │  │  │  3. Sort by similarity distance      │          │     │       │
│  │  │  │  4. Return top 5 matches             │          │     │       │
│  │  │  └──────────────────────────────────────┘          │     │       │
│  │  │                                                      │     │       │
│  │  │  Results:                                            │     │       │
│  │  │  [                                                   │     │       │
│  │  │    {                                                 │     │       │
│  │  │      document: "Course ... Lesson 5 content:        │     │       │
│  │  │                 Prompt caching is a feature...",    │     │       │
│  │  │      metadata: {                                     │     │       │
│  │  │        course_title: "Building Towards...",         │     │       │
│  │  │        lesson_number: 5,                            │     │       │
│  │  │        chunk_index: 142                             │     │       │
│  │  │      },                                              │     │       │
│  │  │      distance: 0.23  # Lower = more similar         │     │       │
│  │  │    },                                                │     │       │
│  │  │    ... 4 more results                               │     │       │
│  │  │  ]                                                   │     │       │
│  │  └─────────────────────────────────────────────────────┘     │       │
│  └────────────────────────┬───────────────────────────────────────┘       │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│            PHASE 7: FORMAT RESULTS (search_tools.py)                     │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  CourseSearchTool._format_results(results)                │           │
│  │                                                            │           │
│  │  For each result:                                         │           │
│  │    1. Extract metadata (course, lesson)                   │           │
│  │    2. Build header: "[Course - Lesson N]"                 │           │
│  │    3. Track source for UI                                 │           │
│  │    4. Combine: header + document                          │           │
│  │                                                            │           │
│  │  Output String:                                            │           │
│  │  ┌────────────────────────────────────────────────┐      │           │
│  │  │ [Building Towards Computer Use - Lesson 5]     │      │           │
│  │  │ Prompt caching is a feature that helps         │      │           │
│  │  │ optimize API usage by caching consistent       │      │           │
│  │  │ prefix elements...                             │      │           │
│  │  │                                                 │      │           │
│  │  │ [Building Towards Computer Use - Lesson 5]     │      │           │
│  │  │ Cache read tokens are 90% cheaper than         │      │           │
│  │  │ uncached input tokens...                       │      │           │
│  │  │                                                 │      │           │
│  │  │ ... (3 more chunks)                            │      │           │
│  │  └────────────────────────────────────────────────┘      │           │
│  │                                                            │           │
│  │  self.last_sources = [                                    │           │
│  │    "Building Towards Computer Use - Lesson 5",            │           │
│  │    "Building Towards Computer Use - Lesson 5",            │           │
│  │    ...                                                     │           │
│  │  ]                                                         │           │
│  └────────────────────────┬───────────────────────────────────┘           │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            │ Return formatted string
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│          PHASE 8: BUILD TOOL RESULT (ai_generator.py)                    │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  tool_results = [{                                        │           │
│  │    "type": "tool_result",                                 │           │
│  │    "tool_use_id": "toolu_abc123",  # Must match request  │           │
│  │    "content": "[Building Towards... formatted results"    │           │
│  │  }]                                                        │           │
│  │                                                            │           │
│  │  messages.append({"role": "user", "content": tool_results})│           │
│  └────────────────────────┬───────────────────────────────────┘           │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            │ API Call #2
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  ANTHROPIC CLAUDE API (Second Call)                      │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  Messages sent to Claude:                                 │           │
│  │  [                                                         │           │
│  │    {"role":"user", "content":"What is prompt caching?"},  │           │
│  │    {"role":"assistant", "content":[                       │           │
│  │      {type:"text", text:"Let me search..."},              │           │
│  │      {type:"tool_use", name:"search_...", input:{...}}    │           │
│  │    ]},                                                     │           │
│  │    {"role":"user", "content":[                            │           │
│  │      {type:"tool_result", tool_use_id:"toolu_abc123",     │           │
│  │       content:"[Building Towards... results]"}            │           │
│  │    ]}                                                      │           │
│  │  ]                                                         │           │
│  │                                                            │           │
│  │  Parameters:                                               │           │
│  │  - NO tools this time (prevent infinite loop)             │           │
│  │  - Same system prompt + history                           │           │
│  │                                                            │           │
│  │  Claude synthesizes final answer:                         │           │
│  │  ┌────────────────────────────────────────────────┐      │           │
│  │  │ "Prompt caching is a feature that helps        │      │           │
│  │  │  optimize API usage by allowing you to cache   │      │           │
│  │  │  consistent prefix elements of your prompts.   │      │           │
│  │  │  It can reduce costs by up to 90% and latency  │      │           │
│  │  │  by up to 85% for repetitive tasks. Cached     │      │           │
│  │  │  tokens cost 90% less than uncached tokens,    │      │           │
│  │  │  with a 5-minute TTL that resets on each read."│      │           │
│  │  └────────────────────────────────────────────────┘      │           │
│  │                                                            │           │
│  │  Response: {                                               │           │
│  │    stop_reason: "end_turn",                               │           │
│  │    content: [{type:"text", text:"Prompt caching..."}]     │           │
│  │  }                                                         │           │
│  └────────────────────────┬───────────────────────────────────┘           │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            │ Return response.content[0].text
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PHASE 9: EXTRACT SOURCES (rag_system.py)                    │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  # AI generator returns final answer                      │           │
│  │  response = "Prompt caching is a feature..."              │           │
│  │                                                            │           │
│  │  # Get sources from tool manager                          │           │
│  │  sources = self.tool_manager.get_last_sources()           │           │
│  │  # Returns: ["Building Towards Computer Use - Lesson 5"]  │           │
│  │                                                            │           │
│  │  # Clean up for next query                                │           │
│  │  self.tool_manager.reset_sources()                        │           │
│  └────────────────────────┬───────────────────────────────────┘           │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│           PHASE 10: UPDATE SESSION (session_manager.py)                  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  self.session_manager.add_exchange(                       │           │
│  │    session_id="session_1",                                │           │
│  │    user_message="What is prompt caching?",                │           │
│  │    assistant_message="Prompt caching is a feature..."     │           │
│  │  )                                                         │           │
│  │                                                            │           │
│  │  Sessions["session_1"] = [                                │           │
│  │    Message(role="user", content="prev question"),         │           │
│  │    Message(role="assistant", content="prev answer"),      │           │
│  │    Message(role="user", content="What is prompt..."),     │           │
│  │    Message(role="assistant", content="Prompt caching...") │           │
│  │  ]                                                         │           │
│  │                                                            │           │
│  │  # Keep only last 4 messages (2 exchanges)                │           │
│  └────────────────────────┬───────────────────────────────────┘           │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            │ return (response, sources)
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 PHASE 11: API RESPONSE (app.py)                          │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  return QueryResponse(                                    │           │
│  │    answer=answer,                                         │           │
│  │    sources=sources,                                       │           │
│  │    session_id=session_id                                  │           │
│  │  )                                                         │           │
│  │                                                            │           │
│  │  HTTP 200 OK                                              │           │
│  │  Content-Type: application/json                           │           │
│  │  {                                                         │           │
│  │    "answer": "Prompt caching is a feature...",            │           │
│  │    "sources": [                                            │           │
│  │      "Building Towards Computer Use - Lesson 5"           │           │
│  │    ],                                                      │           │
│  │    "session_id": "session_1"                              │           │
│  │  }                                                         │           │
│  └────────────────────────┬───────────────────────────────────┘           │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            │ JSON Response
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PHASE 12: FRONTEND DISPLAY (script.js)                      │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │  // Parse response                                        │           │
│  │  const data = await response.json();                      │           │
│  │                                                            │           │
│  │  // Update session ID                                     │           │
│  │  currentSessionId = data.session_id;                      │           │
│  │                                                            │           │
│  │  // Remove loading animation                              │           │
│  │  loadingMessage.remove();                                 │           │
│  │                                                            │           │
│  │  // Render message                                        │           │
│  │  addMessage(data.answer, 'assistant', data.sources);      │           │
│  │                                                            │           │
│  │  Steps:                                                    │           │
│  │  1. Convert markdown to HTML (marked.parse)               │           │
│  │  2. Create message div with content                       │           │
│  │  3. Add collapsible sources section                       │           │
│  │  4. Append to chat container                              │           │
│  │  5. Scroll to bottom                                      │           │
│  └────────────────────────┬───────────────────────────────────┘           │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER SEES RESPONSE                                │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │  Assistant                                                  │         │
│  │  ┌────────────────────────────────────────────────────────┐│         │
│  │  │ Prompt caching is a feature that helps optimize API    ││         │
│  │  │ usage by allowing you to cache consistent prefix       ││         │
│  │  │ elements of your prompts. It can reduce costs by up to ││         │
│  │  │ 90% and latency by up to 85% for repetitive tasks.     ││         │
│  │  │ Cached tokens cost 90% less than uncached tokens,      ││         │
│  │  │ with a 5-minute TTL that resets on each read.          ││         │
│  │  └────────────────────────────────────────────────────────┘│         │
│  │  ▼ Sources                                                  │         │
│  │    Building Towards Computer Use - Lesson 5                │         │
│  └────────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components Summary

| Component | File | Purpose |
|-----------|------|---------|
| **Frontend UI** | `script.js`, `index.html` | User interaction, HTTP requests, message rendering |
| **API Layer** | `app.py` | FastAPI endpoints, request/response validation |
| **Orchestrator** | `rag_system.py` | Coordinates all components |
| **Session Manager** | `session_manager.py` | Conversation history (max 2 exchanges) |
| **AI Generator** | `ai_generator.py` | Claude API calls, tool execution handling |
| **Tool Manager** | `search_tools.py` | Tool registration, routing, execution |
| **Vector Store** | `vector_store.py` | ChromaDB interface, semantic search |
| **Document Processor** | `document_processor.py` | Parse & chunk documents (used at startup) |

## Data Flow Timeline

```
Time    Action
────────────────────────────────────────────────────────────────
0ms     User submits query
50ms    → POST /api/query to FastAPI
100ms   → RAG system receives query
150ms   → Retrieve conversation history from SessionManager
200ms   → AI Generator prepares Claude API call
250ms   → First Claude API call (with tools)
1500ms  ← Claude responds with tool_use
1550ms  → Execute search_course_content tool
1600ms  → Vector store performs semantic search in ChromaDB
1750ms  ← ChromaDB returns top 5 chunks
1800ms  → Format results with course/lesson context
1850ms  → Build tool_result message
1900ms  → Second Claude API call (with tool results)
3000ms  ← Claude responds with final answer
3050ms  → Extract sources from tool
3100ms  → Update session history
3150ms  → Return response to FastAPI
3200ms  ← HTTP 200 with JSON response
3250ms  → Frontend parses JSON
3300ms  → Render markdown & sources
3350ms  User sees complete response
```

## Architecture Pattern

This system implements:
- **Agentic RAG**: AI decides when to search
- **Tool-based workflow**: Claude controls tool usage
- **Two-phase generation**: Search → Synthesize
- **Stateful sessions**: Conversation context preserved
- **Semantic search**: Embeddings for both courses & content
- **Metadata filtering**: ChromaDB filters by course/lesson
