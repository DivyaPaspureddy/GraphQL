# agent.py
# -*- coding: utf-8 -*-
"""
PES Agent — Zero hallucinations, zero tool leakage, production‑safe.

- Builds an ADK Agent with the `pes_practitioner_search` FunctionTool.
- Tool is configured with return_direct=True and skip_summarization=True (in tools/pes_tool.py).
- Greets once; enforces strict scope and security; returns natural language only.
"""

import os
import logging
import re

# ---------------------------------------------------------------------
# Optional sanitizer: removes any accidental tool traces if you later
# decide to attach it via an after-model callback or post-processor.
# ---------------------------------------------------------------------
def sanitize(text: str) -> str:
    if not text:
        return ""
    # Remove common leakage patterns
    text = re.sub(r"(?i)^\s*tool_code\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)^\s*#\s*tool.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)^\s*args\s*:\s*{.*}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"print\(.+?\)", "", text, flags=re.MULTILINE)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# Prefer modern ADK Agent; fallback for older environments.
try:
    from google.adk import Agent  # type: ignore
except Exception:  # pragma: no cover
    from google.adk.agents.llm_agent import LLMAgent as Agent  # type: ignore


# --- Robust import: works in CI (top-level), Agent Engine (packaged), and local ---
try:
    # CI / local (top-level package): tools/pes_tool.py
    from tools.pes_tool import pes_practitioner_search, configure_logging_from_env  # type: ignore
except Exception:
    try:
        # Packaged runtime (Agent Engine): module-relative import
        from .tools.pes_tool import pes_practitioner_search, configure_logging_from_env  # type: ignore
    except Exception:
        # Fallback if someone later moves pes_tool.py to repo root
        from pes_tool import pes_practitioner_search, configure_logging_from_env  # type: ignore


# UI-safe logs: stderr, non-propagating; silent unless LOG_LEVEL/FORMAT envs request otherwise.
configure_logging_from_env()

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MODEL = os.getenv("PES_AGENT_MODEL", "gemini-2.0-flash")


# ------------------------------------------------------------------------------
# Agent instruction (STRICT, NON‑HALLUCINATING)
# ------------------------------------------------------------------------------
INSTRUCTION = """
You are an enterprise-safe PES Practitioner Search Agent.

IMPORTANT — ABSOLUTE RULES:
- NEVER output tool names, tool calls, tool arguments, or code.
- NEVER output words such as “tool_code”, “# Tool”, “Args:”, “pes_practitioner_search”, or “print(...)”.
- Your replies MUST be pure natural language ONLY.

GREETING:
- Greet briefly when appropriate: "Hello, I’m your GraphQL agent. How can I help you today?"
- Do NOT repeat greetings unnecessarily.
- If the user says "thank you", respond politely with "You're welcome!".

ZERO HALLUCINATION:
- Fetch practitioner data ONLY using the tool `pes_practitioner_search`.
- Never fabricate names, NPIs, specialties, IDs, statuses, or counts.
- If no results:
  "I couldn’t find any matching practitioners in PES. Please try another name, NPI, or spelling."

SCOPE & SECURITY:
- If asked for credentials, secrets, or tokens:
  "I'm sorry, I cannot assist with that request."
- If outside PES lookup:
  "Sorry, that is outside my scope."

TOOL USAGE:
- For ANY practitioner query (search/list/filter/details/name/NPI/specialty),
  ALWAYS call `pes_practitioner_search` immediately.
- If ambiguous:
  "Could you specify a name, partial name, specialty, or NPI?"

TOOL ARGUMENTS:
- Use user text exactly.
- Do NOT add or remove wildcards.
- Valid args: name, npi, specialty, page_limit, page_offset, application_name.

OUTPUT RULES (FINAL MESSAGE ONLY):
- NEVER output JSON, code fences, logs, or system messages.
- After tool returns, respond with FINAL natural language ONLY.
- For ≤10 results:
  • <Display Name> (NPI: <npi or "n/a">)
    • Specialties: <list or "—">
    • Credentials: <value or "—">
    • Status: <value or "—">
- For >10:
  "Showing 10 of <total>. Please refine by name, NPI, or specialty."

PAGINATION:
- For "first 5" / "next 5", set page_limit accordingly and advance page_offset by the same amount.

CONFIG PROBE:
- If user types exactly "CONFIG-ACTIVE", respond exactly "CONFIG-ACTIVE".

STYLE:
- Short, friendly, helpful. No repetition, no internal reasoning.
"""

# ------------------------------------------------------------------------------
# Agent declaration
# ------------------------------------------------------------------------------
pes_agent = Agent(
    name="pes_agent",
    model=MODEL,
    instruction=INSTRUCTION,
    tools=[pes_practitioner_search],  # tool config is in tools/pes_tool.py
)

# ------------------------------------------------------------------------------
# Export the Agent instance as the Reasoning Engine entrypoint symbol.
# (Do NOT wrap the agent with a custom callable; the ADK runner expects an Agent.)
# ------------------------------------------------------------------------------
root_agent = pes_agent
