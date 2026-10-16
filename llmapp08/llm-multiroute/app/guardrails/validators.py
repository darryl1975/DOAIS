"""Custom guardrails-ai Validators for llm-multiroute.

Each validator is a real `guardrails.validators.Validator` registered via
`@register_validator`, so they compose with `guardrails.Guard` exactly like
the hosted validators on the Guardrails Hub (e.g. hub://guardrails/toxic_language).
They are implemented with local regex/heuristics (no hub install / network
call / model download required) so the module works out of the box; swap
any of them for a hub-installed validator later without changing the engine.
"""

import re
from typing import Any

from guardrails.validators import FailResult, PassResult, Validator, register_validator

# ── Prompt injection / jailbreak attempts ──────────────────────────
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a)\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
    re.compile(r"(reveal|show|print|output)\s+(your\s+)?(system\s+prompt|instructions|rules)", re.IGNORECASE),
    re.compile(r"what\s+(are|is)\s+your\s+(system\s+prompt|instructions|rules)", re.IGNORECASE),
    re.compile(r"\bsystem\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
]

# ── Harmful / policy-violating content requests ────────────────────
_HARMFUL_CONTENT_PATTERNS = [
    re.compile(r"how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive)", re.IGNORECASE),
    re.compile(r"(generate|create|write)\s+(malware|virus|ransomware)", re.IGNORECASE),
    re.compile(r"(hack|breach|exploit)\s+(into|a)\s+", re.IGNORECASE),
]

# ── Secrets / credentials that should never be forwarded to a third-party model ──
_SECRET_PATTERNS = {
    "openai_api_key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "generic_api_key": re.compile(r"(api|access)[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}", re.IGNORECASE),
    "bearer_token": re.compile(r"Bearer\s+[A-Za-z0-9\-_.]{20,}"),
    "private_key_block": re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----"),
    "password_assignment": re.compile(r"password\s*[:=]\s*\S{6,}", re.IGNORECASE),
}

# ── PII patterns, mapped to a redaction placeholder ────────────────
_PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "telephone": re.compile(r"\d{8}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b"),
}


@register_validator(name="prompt-injection-detector", data_type="string")
class PromptInjectionDetector(Validator):
    """Flags common prompt-injection / jailbreak phrasing."""

    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        matched = [p.pattern for p in _INJECTION_PATTERNS if p.search(value)]
        if matched:
            return FailResult(error_message=f"Prompt injection patterns matched: {matched}")
        return PassResult()


@register_validator(name="harmful-content-detector", data_type="string")
class HarmfulContentDetector(Validator):
    """Flags requests for clearly harmful content (weapons, malware, hacking)."""

    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        matched = [p.pattern for p in _HARMFUL_CONTENT_PATTERNS if p.search(value)]
        if matched:
            return FailResult(error_message=f"Harmful content patterns matched: {matched}")
        return PassResult()


@register_validator(name="secrets-present-detector", data_type="string")
class SecretsPresentDetector(Validator):
    """Detects API keys/tokens/passwords and redacts them when fixed."""

    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        matched = [name for name, pattern in _SECRET_PATTERNS.items() if pattern.search(value)]
        if matched:
            redacted = value
            for name, pattern in _SECRET_PATTERNS.items():
                redacted = pattern.sub("[REDACTED_SECRET]", redacted)
            return FailResult(
                error_message=f"Potential secret(s) detected: {matched}",
                fix_value=redacted,
            )
        return PassResult()


@register_validator(name="pii-detector", data_type="string")
class PIIDetector(Validator):
    """Detects PII (email/phone/SSN/credit card) and redacts it when fixed."""

    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        matched = [name for name, pattern in _PII_PATTERNS.items() if pattern.search(value)]
        if matched:
            redacted = value
            for name, pattern in _PII_PATTERNS.items():
                redacted = pattern.sub(f"[REDACTED_{name.upper()}]", redacted)
            return FailResult(
                error_message=f"PII detected: {matched}",
                fix_value=redacted,
            )
        return PassResult()
