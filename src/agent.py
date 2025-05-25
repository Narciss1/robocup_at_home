"""brain.py
Hub‑and‑spoke *thinking* engine for the robot.  It wraps the LangGraph
pipeline (Planner + three specialists) into a single **Brain** class
with one public coroutine:

```python
reply = await brain.execute(user_text)
```

The implementation is adapted from the Jupyter notebook provided in the
prompt and keeps identical behaviour while hiding all the scaffolding
inside the class.  `Brain.execute()` returns the *conversation
specialist*'s final spoken reply or a summary of what happened.

Dependencies
~~~~~~~~~~~~
    pip install langgraph langchain-openai

You must also set your OpenAI API key (``OPENAI_API_KEY`` environment
variable) or configure the LangChain client according to your backend.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from dotenv import load_dotenv

load_dotenv(".env")  # Load environment variables from .env file

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """You are the PLANNER in charge of orchestrating a robot with four
specialists: vision, manipulation and conversation.
Figure out a short ordered python list of steps (JSON only!) that the robot
must execute to complete the user request.
Return ONLY a python array
Example:
[
  {"agent": "vision",       "command": "locate the banana",          "parallel": false},
  {"agent": "manipulation", "command": "pick up the banana",           "parallel": false},
  {"agent": "conversation", "command": "tell the user the job is done","parallel": false}
]

Remember to output a python list, otherwise the robot won't understand it.
"""

VISION_PROMPT = "You are the VISION specialist. Use vision tools only to locate objects."
MANIP_PROMPT = "You are the MANIPULATION specialist. Pick up or drop items."
CONV_PROMPT = "You are the CONVERSATION specialist. Speak with the human only. You can only pick one tool at a time."

# ---------------------------------------------------------------------------
# Shared state structure for LangGraph
# ---------------------------------------------------------------------------


@dataclass
class _AgentState:
    messages: List[Any] = field(default_factory=list)
    plan: List[Dict[str, Any]] = field(default_factory=list)
    current: Optional[Dict[str, Any]] = None
    results: List[str] = field(default_factory=list)
    n_executed: int = 0


# ---------------------------------------------------------------------------
# Simple mock tool implementations
# ---------------------------------------------------------------------------


def _cv_detect_object(object_name: str):  # noqa: D401
    """Dummy computer‑vision call returning fake coordinates."""
    print("[CV] locating", object_name)
    return (2, 3, 4)


def _manip_grab(object_name: str, coords):  # noqa: D401
    print("[MANIP] grabbing", object_name, "at", coords)
    return True


@tool
def detect_object(object_name: str):
    """Detect an object and return coordinates (dummy impl)."""
    coords = _cv_detect_object(object_name)
    return {
        "status": "success",
        "message": f"Detected {object_name} at {coords}",
    }


@tool
def grab_object(object_name: str, coords: tuple):
    """Pick up an object at given coordinates (dummy impl)."""
    ok = _manip_grab(object_name, coords)
    return {
        "status": "success" if ok else "failure",
        "message": f"Picked up {object_name}" if ok else f"Failed to pick up {object_name}",
    }


@tool
def speak(text: str):  # noqa: D401
    """Return the text as if spoken – actual TTS happens elsewhere."""
    return {"status": "success", "message": text}


CONV_TOOLS = [speak]
VISION_TOOLS = [detect_object]
MANIP_TOOLS = [grab_object]

# ---------------------------------------------------------------------------
# Brain class
# ---------------------------------------------------------------------------


class Brain:
    """Wrapper around the LangGraph multi‑agent planner/executor."""

    _JSON_RE = re.compile(r"\[[\s\S]*]")

    def __init__(self, planner_model: str = "gpt-4o", worker_model: str = "gpt-4o-mini") -> None:
        # Build planner LLM --------------------------------------------------
        planner_llm = ChatOpenAI(model=planner_model, temperature=0.1).with_config({"system_prompt": PLANNER_PROMPT})

        def _parse_plan(text: str):
            m = self._JSON_RE.search(text)
            if not m:
                raise ValueError("Planner produced no JSON list")
            return json.loads(m.group())

        def planner_node(state: _AgentState):  # noqa: D401
            updates: Dict[str, Any] = {}
            if not state.plan and state.n_executed == 0:
                llm_reply: AIMessage = planner_llm.invoke([PLANNER_PROMPT] + [state.messages[-1]])
                plan = _parse_plan(llm_reply.content)
                updates["plan"] = plan
                updates["messages"] = state.messages + [llm_reply]

            plan = updates.get("plan", state.plan)
            if plan:
                step = plan.pop(0)
                updates.update({"plan": plan, "current": step})
                return Command(goto=step["agent"], update=updates)

            # summary = "\n".join(state.results) or "Nothing executed."
            final_msg = AIMessage(content=f"{state.results[-1]}")
            updates["messages"] = state.messages + [final_msg]
            return Command(goto=END, update=updates)

        # Specialist factory -----------------------------------------------
        def make_specialist(name: str, prompt: str, tools: list):
            base_llm = ChatOpenAI(model=worker_model, temperature=0.1).with_config({"system_prompt": prompt})
            llm = base_llm.bind_tools(tools) if tools else base_llm
            tool_node = ToolNode(tools) if tools else None

            def _node(state: _AgentState):  # noqa: D401
                cmd_text = "\nNew message:\n".join(["Passed results:\n"] + state.results + [state.current["command"]])
                ai_msg: AIMessage = llm.invoke(cmd_text)
                msgs = state.messages + [ai_msg]

                success, result = False, ""
                if ai_msg.tool_calls and tool_node:
                    try:
                        out = tool_node.invoke({"messages": [ai_msg]})
                        tool_msgs = out["messages"]
                        msgs.extend(tool_msgs)
                        result = tool_msgs[-1].content
                        success = "failed" not in result.lower()
                    except Exception as exc:  # noqa: BLE001
                        result = f"tool error: {exc}"

                if success:
                    results = state.results + [f"{name}: {result}"]
                    plan = state.plan
                    n_executed = state.n_executed + 1
                else:
                    results = state.results
                    plan = []
                    msgs.append(AIMessage(content=f"{name} FAILED: {result}"))
                    n_executed = state.n_executed

                update = {"messages": msgs, "results": results, "plan": plan, "n_executed": n_executed}
                return Command(goto="planning", update=update)

            return _node

        # Build graph -------------------------------------------------------
        builder = StateGraph(_AgentState)
        builder.add_node("planning", planner_node)
        builder.add_node("vision", make_specialist("vision", VISION_PROMPT, VISION_TOOLS))
        builder.add_node("manipulation", make_specialist("manipulation", MANIP_PROMPT, MANIP_TOOLS))
        builder.add_node("conversation", make_specialist("conversation", CONV_PROMPT, CONV_TOOLS))
        builder.add_edge(START, "planning")
        self._graph = builder.compile()

    # ------------------------------------------------------------------
    async def execute(self, text: str) -> str:  # noqa: D401
        """Run the agent graph for *one* user request and return summary."""
        # LangGraph is synchronous; run in thread so *Brain* can be awaited.
        def _run():
            res = self._graph.invoke({"messages": [("human", text)]})
            res = res["results"][-1]
            # find what's between { and } inclusive
            match = re.search(r"\{(.*?)\}", res)
            if match:
                res = match.group(0)
            res = json.loads(res)["message"] if res else "No result"

            return res

        import asyncio  # lazy import to avoid at top level

        return await asyncio.to_thread(_run)
