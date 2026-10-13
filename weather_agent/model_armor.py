"""Google Model Armor Template & Guardrails Module.

Provides prompt injection, jailbreak detection, and harmful content filtering
modeled after Google Cloud Model Armor.

This module acts as the inspection engine for ADK callbacks to ensure that
inappropriate or malicious queries (e.g., how to create a bomb, system prompt overrides,
or jailbreak exploits) are intercepted before reaching the language model.
"""

from typing import Dict, Any, List, Optional
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger("weather_agent.model_armor")

# Exact response specified for policy violations:
BLOCKED_RESPONSE_TEXT = "Sorry, I cannot answer that.  Your question violates our guidlines."


@dataclass
class ModelArmorResult:
    """Verdict result from Model Armor template screening."""
    is_blocked: bool
    filter_name: Optional[str] = None
    violation_category: Optional[str] = None
    reason: Optional[str] = None
    blocked_message: str = BLOCKED_RESPONSE_TEXT


class ModelArmorTemplate:
    """Model Armor screening template for user prompts and model responses.

    Configured with inspection rules for:
    1. Prompt Injection & Jailbreak (PI/JB) detection.
    2. Harmful & dangerous content (e.g., explosives, bombs, violence, cyber attacks).
    """

    def __init__(self, blocked_message: str = BLOCKED_RESPONSE_TEXT):
        self.blocked_message = blocked_message

        # 1. Prompt Injection and Jailbreak patterns
        self.injection_patterns: List[re.Pattern] = [
            # System prompt / directive overrides
            re.compile(
                r"\b(?:ignore|disregard|forget|override|bypass|clear|drop)\b.*?\b(?:instructions|directives|rules|system\s+prompt|guardrails|safety\s+filters|guidelines)\b",
                re.IGNORECASE
            ),
            # Jailbreak / Developer / DAN persona adoption
            re.compile(
                r"\b(?:jailbreak|dan\s+mode|developer\s+mode|unrestricted\s+mode|god\s+mode|evil\s+mode)\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\b(?:act|pretend|behave)\s+as\s+(?:an?\s+)?(?:unfiltered|unrestricted|uncensored|evil|jailbroken)\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\b(?:you\s+have\s+no\s+(?:rules|restrictions|limits|guidelines))\b",
                re.IGNORECASE
            ),
            # Secret / System prompt exfiltration
            re.compile(
                r"\b(?:reveal|show|display|print|leak|repeat|output)\b.*?\b(?:system\s+prompt|secret\s+instructions|hidden\s+rules|internal\s+prompt)\b",
                re.IGNORECASE
            ),
        ]

        # 2. Harmful & Dangerous content patterns (weapons, bombs, explosives, attacks)
        self.harmful_patterns: List[re.Pattern] = [
            # Explosives and bombs
            re.compile(
                r"\b(?:how\s+(?:to|do\s+i|can\s+i|does\s+one)\s+)?(?:make|create|build|construct|assemble|detonate|manufacture|prepare)\b.*?\b(?:bombs?|explosives?|ieds?|pipe\s*bombs?|dynamite|grenades?|c4|molotov)\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\b(?:bombs?|pipe\s*bombs?|car\s*bombs?|suicide\s*bombs?|fertilizer\s*bombs?|dirty\s*bombs?|explosives?|detonators?)\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\b(?:how\s+(?:to|do\s+i|can\s+i|does\s+one)\s+)?(?:make|synthesize|manufacture)\b.*?\b(?:poisons?|nerve\s*agents?|ricin|anthrax|sarin|cyanide)\b",
                re.IGNORECASE
            ),
            # Cyber attacks / Malware
            re.compile(
                r"\b(?:how\s+(?:to|do\s+i|can\s+i|does\s+one)\s+)?(?:write|code|create|build)\b.*?\b(?:malware|ransomware|keyloggers?|trojans?|rootkits?|ddos\s+botnets?)\b",
                re.IGNORECASE
            ),
            # Violent crimes / Weapons manufacturing
            re.compile(
                r"\b(?:how\s+(?:to|do\s+i|can\s+i|does\s+one)\s+)?(?:kill|murder|assassinate|attack)\b.*?\b(?:someone|people|person|president|crowds?)\b",
                re.IGNORECASE
            ),
            re.compile(
                r"\b(?:how\s+(?:to|do\s+i|can\s+i|does\s+one)\s+)?(?:make|build|3d\s*print)\b.*?\b(?:guns?|firearms?|silencers?|ghost\s*guns?)\b",
                re.IGNORECASE
            ),
        ]

    def sanitize_user_prompt(self, text: str) -> ModelArmorResult:
        """Screens a user prompt against the Model Armor template filters.

        Args:
            text (str): The user's input prompt text.

        Returns:
            ModelArmorResult: Verdict indicating whether the prompt was blocked
                or allowed, including violation category and reason if blocked.
        """
        if not text or not text.strip():
            return ModelArmorResult(is_blocked=False)

        clean_text = text.strip()

        # Check harmful content filter (e.g. bomb creation, weapons)
        for pattern in self.harmful_patterns:
            match = pattern.search(clean_text)
            if match:
                logger.warning(
                    "Model Armor Harmful Content Match: '%s' matched '%s'",
                    match.group(0), clean_text
                )
                return ModelArmorResult(
                    is_blocked=True,
                    filter_name="harmful_content_filter",
                    violation_category="Dangerous & Harmful Content (Explosives/Weapons)",
                    reason=f"Matched prohibited phrase: '{match.group(0)}'",
                    blocked_message=self.blocked_message
                )

        # Check prompt injection and jailbreak filter
        for pattern in self.injection_patterns:
            match = pattern.search(clean_text)
            if match:
                logger.warning(
                    "Model Armor Prompt Injection Match: '%s' matched '%s'",
                    match.group(0), clean_text
                )
                return ModelArmorResult(
                    is_blocked=True,
                    filter_name="prompt_injection_and_jailbreak_filter",
                    violation_category="Prompt Injection & Jailbreak Attempt",
                    reason=f"Matched injection attempt: '{match.group(0)}'",
                    blocked_message=self.blocked_message
                )

        return ModelArmorResult(is_blocked=False)


# Default global template instance
default_model_armor = ModelArmorTemplate()


def screen_prompt(text: str) -> ModelArmorResult:
    """Convenience function to screen a prompt using default Model Armor template."""
    return default_model_armor.sanitize_user_prompt(text)
