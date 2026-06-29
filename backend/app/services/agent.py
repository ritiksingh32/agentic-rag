"""
Agent loop.

The core decision-making logic (Point F.7 in the study guide). Instead
of a fixed retrieve -> generate pipeline, this loop lets the LLM decide
which tool(s) to call, in what order, and how many times — based on
function calling / tool use.

Guardrails built in from the spec:
- MAX_ITERATIONS hard limit, to prevent runaway/infinite tool-call loops
- Per-tool error handling (tools return error strings instead of raising,
  so a failed tool doesn't crash the whole agent turn)
- Conversation memory is handled by the CALLER passing in prior turns,
  not stored here — this module is stateless by design.
"""

import json

from groq import Groq

from app.core.config import settings
from app.services.tools import (
    TOOL_SCHEMAS,
    tool_retrieve_documents,
    tool_web_search,
    tool_compare_documents,
    tool_list_my_documents,
)

groq_client = Groq(api_key=settings.GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

MAX_ITERATIONS = 4

SYSTEM_PROMPT = """You are a helpful research assistant with access to tools.

IMPORTANT — tool selection priority:
1. ALWAYS try retrieve_documents FIRST for any question that could plausibly
   be answered by the user's own uploaded documents — this includes general
   topic questions, not just questions that explicitly say "in my document."
   For example, "What is human evolution?" should trigger retrieve_documents,
   because the user likely uploaded a document about that exact topic.
2. If retrieve_documents returns relevant, on-topic source chunks, you MUST
   answer using ONLY those chunks and STOP — do not also call web_search.
   Calling web_search after already finding good document results is WRONG
   and wastes resources. Only call web_search if retrieve_documents returned
   NO relevant results, or the question is clearly time-sensitive/current
   (e.g. "today", "latest", "current price", recent news).
3. Use compare_documents when the user explicitly asks to compare two of
   their documents.
4. Use list_my_documents when the user asks what documents they have, or
   you need a doc_id.

General rules:
- Once you have relevant results from ANY tool, stop calling more tools and
  answer immediately. Do not call multiple tools "just to be thorough."
- Always cite sources using the [1], [2] style numbers shown in tool results.
- If the document results are relevant, your answer MUST cite them as [1],
  [2], etc. and must NOT cite web sources in the same answer.
- If no tool result answers the question, say so honestly instead of guessing.
"""


def _execute_tool(tool_name: str, arguments: dict, user_id: str, list_documents_fn) -> str:
    """
    Routes a tool call (by name) to its actual implementation.
    Every branch is wrapped so a bad/missing argument can't crash the
    whole agent turn — it just returns an error string the LLM can see
    and react to.
    """
    try:
        if tool_name == "retrieve_documents":
            return tool_retrieve_documents(user_id=user_id, query=arguments["query"])

        elif tool_name == "web_search":
            return tool_web_search(query=arguments["query"])

        elif tool_name == "compare_documents":
            return tool_compare_documents(
                user_id=user_id,
                query=arguments["query"],
                doc_id_a=arguments["doc_id_a"],
                doc_id_b=arguments["doc_id_b"],
            )

        elif tool_name == "list_my_documents":
            return tool_list_my_documents(user_id=user_id, list_documents_fn=list_documents_fn)

        else:
            return f"Unknown tool requested: {tool_name}"

    except KeyError as e:
        return f"Tool call failed — missing required argument: {e}"
    except Exception as e:
        return f"Tool call failed with an unexpected error: {e}"


def run_agent(
    user_id: str,
    question: str,
    conversation_history: list[dict] | None = None,
    list_documents_fn=None,
) -> dict:
    """
    Runs the agent loop for one user turn.

    Args:
        user_id: the current user, passed through to every tool call
                 for multi-tenant isolation
        question: the user's new question
        conversation_history: prior turns, e.g.
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            passed in by the caller so follow-up questions have context
        list_documents_fn: injected function for the list_my_documents
            tool (wired up once the documents DB layer exists)

    Returns:
        {
            "answer": "...",
            "reasoning_trace": [ {"tool": "...", "arguments": {...}}, ... ],
            "hit_iteration_limit": bool
        }
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    reasoning_trace = []
    hit_iteration_limit = False

    for iteration in range(MAX_ITERATIONS):
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
        )

        message = response.choices[0].message

        # Case 1: the LLM is done reasoning and gave a final answer
        if not message.tool_calls:
            return {
                "answer": message.content,
                "reasoning_trace": reasoning_trace,
                "hit_iteration_limit": False,
            }

        # Case 2: the LLM wants to call one or more tools
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            reasoning_trace.append({"tool": tool_name, "arguments": arguments})

            result = _execute_tool(tool_name, arguments, user_id, list_documents_fn)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # Loop again — the LLM gets to see tool results and decide what's next

    # If we get here, MAX_ITERATIONS was hit without a final answer.
    # Force one last call with NO tools available, so the model MUST
    # answer using whatever it has gathered so far (the anti-runaway guardrail).
    hit_iteration_limit = True
    final_response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages + [{
            "role": "user",
            "content": "Please give your best final answer now, based on what you've found so far.",
        }],
        temperature=0.2,
    )

    return {
        "answer": final_response.choices[0].message.content,
        "reasoning_trace": reasoning_trace,
        "hit_iteration_limit": hit_iteration_limit,
    }