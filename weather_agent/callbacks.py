"""Lifecycle Callbacks & Conversation Logging for Google ADK Weather Agent.

Implements:
1. Conversation logging (to both Python logger and conversation.log file).
2. Input moderation & guardrails powered by Google Model Armor template.
3. Chained before_model_callback and after_model_callback for ADK agents.
"""

from typing import Optional, Any
import logging
import os
from datetime import datetime
from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from weather_agent.model_armor import screen_prompt, BLOCKED_RESPONSE_TEXT

logger = logging.getLogger("weather_agent.callbacks")

# Path to the persistent conversation log file
CONVERSATION_LOG_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "conversation.log")
)


class ConversationLogger:
    """Manages persistent logging of user and model conversations."""

    @staticmethod
    def _write_entry(entry: str):
        """Appends a log entry to the conversation.log file."""
        try:
            with open(CONVERSATION_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception as e:
            logger.error("Failed to write to conversation.log: %s", e)

    @classmethod
    def log_user(cls, agent_name: str, session_id: str, message: str):
        """Logs a user query."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [USER]    (Agent: {agent_name} | Session: {session_id}) -> {message}"
        logger.info(entry)
        cls._write_entry(entry)

    @classmethod
    def log_model(cls, agent_name: str, session_id: str, response: str):
        """Logs a model response."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [MODEL]   (Agent: {agent_name} | Session: {session_id}) -> {response}"
        logger.info(entry)
        cls._write_entry(entry)

    @classmethod
    def log_blocked(cls, agent_name: str, session_id: str, user_prompt: str, category: str, reason: str, blocked_response: str):
        """Logs a blocked inappropriate query."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"[{timestamp}] [BLOCKED] (Agent: {agent_name} | Session: {session_id}) "
            f"Category: {category} | Reason: {reason}\n"
            f"           USER INPUT: {user_prompt}\n"
            f"           RESPONSE:   {blocked_response}"
        )
        logger.warning(entry)
        cls._write_entry(entry)


def _extract_last_user_text(llm_request: LlmRequest) -> Optional[str]:
    """Helper to extract the most recent user turn text from an LlmRequest."""
    if not llm_request.contents:
        return None
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            text_parts = [p.text for p in content.parts if hasattr(p, "text") and p.text]
            if text_parts:
                return " ".join(text_parts).strip()
    return None


def _extract_model_text(llm_response: LlmResponse) -> Optional[str]:
    """Helper to extract text from an LlmResponse."""
    if llm_response.content and llm_response.content.parts:
        text_parts = [p.text for p in llm_response.content.parts if hasattr(p, "text") and p.text]
        if text_parts:
            return " ".join(text_parts).strip()
    return None


def moderate_user_prompt(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Validates user input against Model Armor template before sending to the model.

    If the query is inappropriate (e.g., asking how to create a bomb, prompt injection,
    or jailbreak attempt), returns a blocked response and prevents the LLM call.
    """
    user_text = _extract_last_user_text(llm_request)
    if not user_text:
        return None

    agent_name = getattr(callback_context, "agent_name", "weather_agent")
    session_id = getattr(callback_context, "session_id", "unknown_session")

    # Screen using Model Armor template
    verdict = screen_prompt(user_text)
    if verdict.is_blocked:
        # Log security violation
        ConversationLogger.log_blocked(
            agent_name=agent_name,
            session_id=str(session_id),
            user_prompt=user_text,
            category=verdict.violation_category or "Violation",
            reason=verdict.reason or "Policy violation",
            blocked_response=verdict.blocked_message
        )

        # Return blocked LlmResponse immediately to short-circuit model execution
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=verdict.blocked_message)],
            ),
            custom_metadata={
                "model_armor_blocked": True,
                "violation_category": verdict.violation_category,
                "reason": verdict.reason
            }
        )

    return None


def log_user_prompt(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Callback to log approved user prompts."""
    user_text = _extract_last_user_text(llm_request)
    if user_text:
        agent_name = getattr(callback_context, "agent_name", "weather_agent")
        session_id = getattr(callback_context, "session_id", "unknown_session")
        ConversationLogger.log_user(
            agent_name=agent_name,
            session_id=str(session_id),
            message=user_text
        )
    return None


def chained_before_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Chains moderation validation and logging before sending the request to the model."""
    # 1. Moderation check using Model Armor template
    moderation_result = moderate_user_prompt(callback_context, llm_request)
    if moderation_result is not None:
        return moderation_result  # STOP: request was inappropriate / blocked

    # 2. Log approved user prompt
    log_user_prompt(callback_context, llm_request)
    return None  # Proceed to model


def log_model_response(
    callback_context: CallbackContext,
    llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Callback to log model responses."""
    model_text = _extract_model_text(llm_response)
    if model_text:
        agent_name = getattr(callback_context, "agent_name", "weather_agent")
        session_id = getattr(callback_context, "session_id", "unknown_session")
        ConversationLogger.log_model(
            agent_name=agent_name,
            session_id=str(session_id),
            response=model_text
        )
    return None
