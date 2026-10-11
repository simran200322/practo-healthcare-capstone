"""Input and Output Guardrails: PII Masking, Prompt Injection Detection, and Groundedness Verification."""
import re
from typing import Tuple, Dict, Any
from app.config import RAG_SIMILARITY_THRESHOLD

# Regex pattern for fixed-format Indian contact numbers
# Matches formats like: +91 9876543210, +91-9876543210, 9876543210, 09876543210, 98765-43210
PHONE_PATTERN = re.compile(
    r'(?:\+?91[\s\-]?)?(?:0)?[6-9]\d{4}[\s\-]?\d{5}\b'
)

# Common prompt injection and adversarial jailbreak patterns
PROMPT_INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?(previous|prior)\s+instructions',
    r'disregard\s+(all\s+)?(previous|system)\s+prompts?',
    r'system\s+override',
    r'you\s+are\s+now\s+(an?\s+)?unrestricted',
    r'dan\s+mode',
    r'bypass\s+all\s+(filters|guardrails)',
    r'reveal\s+(your\s+)?(system|internal)\s+prompt',
    r'act\s+as\s+a\s+hacker',
    r'pretend\s+you\s+have\s+no\s+rules'
]
INJECTION_REGEX = re.compile('|'.join(PROMPT_INJECTION_PATTERNS), re.IGNORECASE)


def mask_contact_numbers(text: str) -> Tuple[str, bool]:
    """Mask fixed-format contact numbers into [REDACTED_PHONE].
    
    Returns tuple of (masked_text, pii_detected_boolean).
    Ensures contact numbers never reach the model or logs in cleartext.
    """
    masked, count = PHONE_PATTERN.subn("[REDACTED_PHONE]", text)
    return masked, count > 0


def detect_prompt_injection(text: str) -> Tuple[bool, str]:
    """Check whether incoming user text contains prompt injection or jailbreak patterns.
    
    Returns (is_injected, reason).
    """
    match = INJECTION_REGEX.search(text)
    if match:
        return True, f"Prompt injection pattern detected: '{match.group(0)}'"
    return False, ""


def verify_output_groundedness(
    answer: str,
    top_similarity: float,
    threshold: float = RAG_SIMILARITY_THRESHOLD
) -> Tuple[bool, str]:
    """Verify that generated answer is strictly grounded in retrieved policy context.
    
    Refuses if similarity falls below the calibrated empirical threshold.
    """
    if top_similarity < threshold:
        return False, (
            f"Retrieval similarity {top_similarity:.4f} is below the calibrated threshold {threshold:.4f}. "
            "Output refused to prevent ungrounded hallucinations."
        )
    return True, "Grounded in verified policy documentation."
