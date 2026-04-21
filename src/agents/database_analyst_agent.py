"""
Database Analyst Agent 
"""

import json
import time

from src.utils.logging_config import get_logger
from src.config.containers import get_llm_provider, get_async_database, get_embedding_provider
from src.domain.models import Message, ChatRequest

logger = get_logger(__name__)

ITEMS_TABLE_SCHEMA = """
TABLE: items
  - id              SERIAL PRIMARY KEY
  - receipt_id      INTEGER NOT NULL
  - item_name       TEXT NOT NULL
  - quantity        REAL NOT NULL
  - unit_price      REAL NOT NULL
  - total_price     REAL NOT NULL
  - purchase_date   TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
  - item_name_embedding  vector(768)
"""

ANALYST_SYSTEM_PROMPT = f"""You are a database analyst. Answer spending questions by using your tools.

Schema:
{ITEMS_TABLE_SCHEMA}

QUESTION TYPES and how to handle each:

1. ITEM LOOKUP — "How much did I spend on rice?"
   → Call search_similar_items first to find the exact DB spelling.
   → Then execute_sql_query using those exact names with ILIKE.

2. AGGREGATION — "What is my total spending this month?" / "How much did I spend last week?"
   → Skip search_similar_items. Go directly to execute_sql_query.
   → Use date functions: DATE_TRUNC, CURRENT_DATE, intervals.
   → Example: WHERE purchase_date >= DATE_TRUNC('month', CURRENT_DATE)

3. TOP-N / RANKING — "What are my 5 most expensive items?" / "Which item do I buy most often?"
   → Skip search_similar_items. Go directly to execute_sql_query.
   → Use ORDER BY + LIMIT.

4. COMPARISON — "Did I spend more on vegetables or meat?"
   → Call search_similar_items ONCE for each term being compared.
   → Then execute a single SQL with CASE or GROUP BY to compare them.

5. TREND / TIME-SERIES — "How has my grocery spending changed over the past 3 months?"
   → Skip search_similar_items. Go directly to execute_sql_query.
   → Use DATE_TRUNC('month', purchase_date) and GROUP BY month.

6. CATEGORY BROWSE — "Show me all dairy items I bought."
   → Call search_similar_items with the category name to find related items.
   → Then execute_sql_query using those item names.

RULES:
- Only SELECT queries. Never INSERT, UPDATE, DELETE.
- No fabricated data. Only return what the DB contains.
- For date ranges, always use parameterised intervals, not hardcoded dates.
- Return amounts formatted as currency where relevant.
- If no results found, say so clearly — do not guess.
"""

ANALYST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_similar_items",
            "description": (
                "Search for items whose names are semantically similar to a query. "
                "Returns actual item names and similarity scores. "
                "Use BEFORE writing SQL when the question references a specific item or category. "
                "For comparisons, call this once per term being compared. "
                "Skip this entirely for pure aggregation, top-N, or time-series questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The item name or category from the user's question.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return. Default 5.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": (
                "Execute a read-only SQL SELECT query against the 'items' table. "
                "Supports all SELECT features: GROUP BY, ORDER BY, LIMIT, date functions, "
                "SUM, COUNT, AVG, MIN, MAX, CASE expressions, DATE_TRUNC, intervals. "
                "Use exact item names from search_similar_items when querying by name, "
                "or write aggregation/trend queries directly without a prior search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL SELECT query to execute.",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-line description of what this query computes, for logging.",
                    },
                },
                "required": ["sql", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_spending_summary",
            "description": (
                "Get a pre-built spending summary for a time period. "
                "Use for broad questions like 'how much did I spend this week/month/year' "
                "or 'give me a summary of my recent spending'. "
                "Faster than writing custom SQL for common time-range aggregations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["today", "this_week", "this_month", "last_month", "this_year", "last_7_days", "last_30_days"],
                        "description": "The time period to summarise.",
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["none", "day", "week", "month", "item_name"],
                        "description": "How to break down the results. Default is 'none' (single total).",
                        "default": "none",
                    },
                },
                "required": ["period"],
            },
        },
    },
]


PERIOD_SQL = {
    "today":       "purchase_date::date = CURRENT_DATE",
    "this_week":   "purchase_date >= DATE_TRUNC('week', CURRENT_DATE)",
    "this_month":  "purchase_date >= DATE_TRUNC('month', CURRENT_DATE)",
    "last_month":  (
        "purchase_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') "
        "AND purchase_date < DATE_TRUNC('month', CURRENT_DATE)"
    ),
    "this_year":   "purchase_date >= DATE_TRUNC('year', CURRENT_DATE)",
    "last_7_days": "purchase_date >= CURRENT_DATE - INTERVAL '7 days'",
    "last_30_days":"purchase_date >= CURRENT_DATE - INTERVAL '30 days'",
}

GROUP_BY_SQL = {
    "none":      ("SUM(total_price) AS total_spent, COUNT(*) AS item_count", ""),
    "day":       ("DATE_TRUNC('day', purchase_date) AS period, SUM(total_price) AS total_spent", "GROUP BY period ORDER BY period"),
    "week":      ("DATE_TRUNC('week', purchase_date) AS period, SUM(total_price) AS total_spent", "GROUP BY period ORDER BY period"),
    "month":     ("DATE_TRUNC('month', purchase_date) AS period, SUM(total_price) AS total_spent", "GROUP BY period ORDER BY period"),
    "item_name": ("item_name, SUM(total_price) AS total_spent, COUNT(*) AS times_bought", "GROUP BY item_name ORDER BY total_spent DESC"),
}


async def _generate_query_embedding(text: str):
    start_time = time.time()
    try:
        embedding_provider = get_embedding_provider()
        embedding = await embedding_provider.generate_embedding(text)
        logger.debug(f"Embedding generated in {time.time() - start_time:.3f}s")
        return embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return [0.0] * 768
    
async def _handle_search_similar_items(db, args: dict, start_time: float) -> str:
    query = args.get("query", "")
    limit = args.get("limit", 5)

    embedding = await _generate_query_embedding(query)
    results = await db.search_similar_items(query_embedding=embedding, limit=limit)
    elapsed = time.time() - start_time

    if not results:
        logger.info(f"search_similar_items: no results for '{query}' ({elapsed:.3f}s)")
        return json.dumps({"status": "no_results", "message": f"No items similar to '{query}' found."})

    logger.info(f"search_similar_items: {len(results)} results for '{query}' ({elapsed:.3f}s)")
    return json.dumps({"status": "success", "similar_items": results})


async def _handle_execute_sql_query(db, args: dict, start_time: float) -> str:
    sql = args.get("sql", "")
    description = args.get("description", "unnamed query")

    logger.info(f"execute_sql_query [{description}]: {sql[:120]}...")
    rows = await db.execute_read_query(sql)
    elapsed = time.time() - start_time

    if not rows:
        return json.dumps({"status": "no_results", "message": "Query returned no results.", "sql": sql})

    logger.info(f"execute_sql_query: {len(rows)} rows ({elapsed:.3f}s)")
    return json.dumps({"status": "success", "row_count": len(rows), "results": rows, "sql": sql}, default=str)


async def _handle_get_spending_summary(db, args: dict, start_time: float) -> str:
    period   = args.get("period", "this_month")
    group_by = args.get("group_by", "none")

    where_clause = PERIOD_SQL.get(period)
    if not where_clause:
        return json.dumps({"status": "error", "message": f"Unknown period: {period}"})

    select_cols, group_order = GROUP_BY_SQL.get(group_by, GROUP_BY_SQL["none"])
    sql = f"SELECT {select_cols} FROM items WHERE {where_clause} {group_order}".strip()

    logger.info(f"get_spending_summary: period={period}, group_by={group_by}")
    rows = await db.execute_read_query(sql)
    elapsed = time.time() - start_time

    if not rows:
        return json.dumps({"status": "no_results", "message": f"No spending data found for {period}."})

    return json.dumps({
        "status": "success", "period": period, "group_by": group_by,
        "row_count": len(rows), "results": rows, "sql": sql,
    }, default=str)


_TOOL_HANDLERS = {
    "search_similar_items":  _handle_search_similar_items,
    "execute_sql_query":     _handle_execute_sql_query,
    "get_spending_summary":  _handle_get_spending_summary,
}

async def _execute_tool(tool_name: str, tool_args: dict) -> str:
    db = get_async_database()
    start_time = time.time()

    handler = _TOOL_HANDLERS.get(tool_name)
    if not handler:
        logger.warning(f"Unknown tool: {tool_name}")
        return json.dumps({"status": "error", "message": f"Unknown tool: {tool_name}"})

    try:
        return await handler(db, tool_args, start_time)
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Tool error ({tool_name}) after {elapsed:.3f}s: {e}")
        return json.dumps({"status": "error", "message": str(e)})


async def ask_analyst(user_question: str) -> str:
    start_time = time.time()
    llm_provider = get_llm_provider()
    logger.info(f"[Analyst] Question: '{user_question}'")

    messages = [
        Message(role="system", content=ANALYST_SYSTEM_PROMPT),
        Message(role="user", content=user_question),
    ]

    # Comparison questions need: search A → search B → SQL → answer = 3 tool calls
    max_iterations = 5

    for iteration in range(max_iterations):
        chat_request = ChatRequest(
            messages=messages,
            model=llm_provider.get_model_name(),
            tools=ANALYST_TOOLS,
            tool_choice="required" if iteration == 0 else "auto",
        )

        response = llm_provider.chat_completion(chat_request)

        if not response.tool_calls:
            total_elapsed = time.time() - start_time
            logger.info(f"[Analyst] Done in {total_elapsed:.2f}s ({iteration + 1} iterations)")
            return response.content or "No data found for your query."

        messages.append(Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls,
        ))

        for tool_call in response.tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            logger.info(f"[Analyst] Tool call: {function_name}({function_args})")

            tool_result = await _execute_tool(function_name, function_args)
            messages.append(Message(
                tool_call_id=tool_call["id"],
                role="tool",
                name=function_name,
                content=tool_result,
            ))

    logger.warning(f"[Analyst] Max iterations reached, summarising...")
    messages.append(Message(role="user", content="Please summarise whatever results you have so far."))
    final_response = llm_provider.chat_completion(ChatRequest(messages=messages, model=llm_provider.get_model_name()))
    return final_response.content or "Could not complete the analysis."