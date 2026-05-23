from __future__ import annotations

import ast
import datetime
import math
import operator
import re
import urllib.parse
from typing import Callable
import requests


#  Safe Math Evaluator

_SAFE_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_SAFE_NAMES = {
    "pi": math.pi, "e": math.e,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
    "log10": math.log10, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "abs": abs, "round": round, "pow": math.pow,
    "ceil": math.ceil, "floor": math.floor,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Non-numeric constant")
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        func = _SAFE_NAMES.get(node.func.id)
        if func is None:
            raise ValueError(f"Unknown function: {node.func.id}")
        args = [_safe_eval(a) for a in node.args]
        return func(*args)
    if isinstance(node, ast.Name):
        val = _SAFE_NAMES.get(node.id)
        if val is None:
            raise ValueError(f"Unknown name: {node.id}")
        return val
    raise ValueError(f"Unsupported node type: {type(node).__name__}")


#  Individual tools

def calculator(expression: str) -> str:
    """Safely evaluate a math expression and return the result."""
    expr = expression.strip()
    # Remove markdown backticks if present
    expr = re.sub(r"`+", "", expr).strip()
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
        # Format nicely
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"Result: {result}"
    except Exception as exc:
        return f"Calculator error: {exc}. Expression: {expr!r}"


def get_datetime(query: str = "") -> str:
    """Return current UTC and local date/time information."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_local = datetime.datetime.now()
    return (
        f"Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"Current local time: {now_local.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Day of week: {now_local.strftime('%A')}\n"
        f"ISO date: {now_local.date().isoformat()}"
    )


def wikipedia_search(query: str) -> str:
    """Fetch a short Wikipedia summary for the query."""
    query = query.strip()
    if not query:
        return "Please provide a search term."
    try:
        encoded = urllib.parse.quote(query)
        url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        )
        resp = requests.get(url, timeout=6, headers={"User-Agent": "AIAssistantEval/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            title = data.get("title", query)
            extract = data.get("extract", "No summary available.")
            # Truncate to ~400 chars
            if len(extract) > 400:
                extract = extract[:400].rsplit(" ", 1)[0] + "…"
            return f"Wikipedia — {title}:\n{extract}"
        elif resp.status_code == 404:
            return f"No Wikipedia article found for: {query!r}"
        else:
            return f"Wikipedia API error: HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        return "Wikipedia search timed out. Please try again."
    except Exception as exc:
        return f"Wikipedia search failed: {exc}"


def weather_stub(location: str) -> str:
    """
    Stub for weather lookup.
    Replace with a real API call (e.g. OpenWeatherMap) by setting
    OPENWEATHER_API_KEY in your environment.
    """
    import os
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    if not api_key:
        return (
            f"[Weather tool] No API key configured. "
            f"Set OPENWEATHER_API_KEY to enable live weather for '{location}'."
        )
    try:
        encoded = urllib.parse.quote(location)
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={encoded}&appid={api_key}&units=metric"
        )
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            d = resp.json()
            desc = d["weather"][0]["description"].capitalize()
            temp = d["main"]["temp"]
            feels = d["main"]["feels_like"]
            humidity = d["main"]["humidity"]
            city = d.get("name", location)
            return (
                f"Weather in {city}: {desc}\n"
                f"Temperature: {temp}°C (feels like {feels}°C)\n"
                f"Humidity: {humidity}%"
            )
        return f"Weather API error: HTTP {resp.status_code}"
    except Exception as exc:
        return f"Weather lookup failed: {exc}"


#  Tool Registry

TOOLS: dict[str, Callable[[str], str]] = {
    "calculator": calculator,
    "datetime": get_datetime,
    "wikipedia": wikipedia_search,
    "weather": weather_stub,
}

TOOL_DESCRIPTIONS = {
    "calculator": (
        "Evaluate a mathematical expression safely. "
        "Supports +, -, *, /, **, %, //, and functions like sqrt, sin, cos, log, abs."
    ),
    "datetime": "Get the current date and time.",
    "wikipedia": "Search Wikipedia for a brief summary of a topic.",
    "weather": "Look up current weather for a city (requires OPENWEATHER_API_KEY).",
}


def detect_and_call_tool(user_message: str) -> str | None:
    """
    Heuristically detect if the user wants a tool result and return it,
    or None if no tool is applicable.

    This lightweight router avoids an extra LLM call for obvious cases.
    """
    msg = user_message.lower()

    # Math expressions
    if re.search(r"[\d\+\-\*\/\^%].*=\s*\?|what is\s+[\d\s\+\-\*\/\(\)]+|calculate|compute|solve", msg):
        # Extract expression
        expr_match = re.search(r"([\d\s\+\-\*\/\(\)\.\^%]+)", user_message)
        if expr_match:
            return calculator(expr_match.group(1).strip())

    # Date / time
    if re.search(r"\b(what(?:'s| is) (the |today'?s? )?date|what time|current time|today'?s date)\b", msg):
        return get_datetime()

    # Wikipedia
    wiki_match = re.search(r"(?:who|what) is (.+?)[\?\.!]?$", msg)
    if wiki_match and len(wiki_match.group(1).split()) <= 6:
        return wikipedia_search(wiki_match.group(1))

    return None


def call_tool(tool_name: str, argument: str) -> str:
    """Call a named tool with an argument string."""
    tool = TOOLS.get(tool_name.lower())
    if tool is None:
        return f"Unknown tool: {tool_name!r}. Available: {list(TOOLS)}"
    return tool(argument)
