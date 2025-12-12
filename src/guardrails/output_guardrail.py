"""
Output Guardrail
Checks system outputs for safety violations.
"""

from typing import Dict, Any, List
import re
from datetime import datetime


class OutputGuardrail:
    """
    Guardrail for checking output safety.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Check for harmful content in responses
    - Verify factual consistency
    - Detect potential misinformation
    - Remove PII (personal identifiable information)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.topic = config.get("system", {}).get("topic", "HCI research")

    def validate(self, response: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate output response.

        Args:
            response: Generated response to validate
            sources: Optional list of sources used (for fact-checking)

        Returns:
            Validation result

        TODO: YOUR CODE HERE
        - Implement validation logic
        - Check for harmful content
        - Check for PII
        - Verify claims against sources
        - Check for bias
        """
        violations = []

        pii_violations = self._check_pii(response)
        violations.extend(pii_violations)

        harmful_violations = self._check_harmful_content(response)
        violations.extend(harmful_violations)

        if sources:
            consistency_violations = self._check_factual_consistency(response, sources)
            violations.extend(consistency_violations)

        bias_violations = self._check_bias(response)
        violations.extend(bias_violations)

        # Only block when medium/high severity violations exist; low severity will annotate but not block
        has_blocking = any(v.get("severity") in ["medium", "high"] for v in violations)

        return {
            "valid": not has_blocking,
            "violations": violations,
            "sanitized_output": self._sanitize(response, violations) if violations else response,
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for personally identifiable information.

        TODO: YOUR CODE HERE Implement comprehensive PII detection
        """
        violations = []

        # Simple regex patterns for common PII
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        }

        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append({
                    "validator": "pii",
                    "pii_type": pii_type,
                    "reason": f"Contains {pii_type}",
                    "severity": "high",
                    "matches": matches
                })

        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for harmful or inappropriate content.

        """
        violations = []

        harmful_keywords = [
            "violent",
            "harmful",
            "dangerous",
            "weapon",
            "attack",
            "kill",
            "exploit",
            "self-harm",
        ]
        for keyword in harmful_keywords:
            if keyword in text.lower():
                violations.append({
                    "validator": "harmful_content",
                    "reason": f"May contain harmful content: {keyword}",
                    "severity": "medium"
                })

        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check if response is consistent with sources.

        TODO: YOUR CODE HERE Implement fact-checking logic
        This could use LLM-based verification
        """
        violations = []

        # Lightweight consistency check: ensure at least one source is referenced
        if sources:
            has_citation = any(url in response for url in [s.get("url", "") for s in sources if s])
            if not has_citation:
                violations.append({
                    "validator": "citations",
                    "reason": "Response lacks explicit references to provided sources",
                    "severity": "low",
                })

        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for biased language.

        """
        violations = []
        biased_phrases = [
            "obviously",
            "clearly",
            "everyone knows",
        ]
        for phrase in biased_phrases:
            if phrase in text.lower():
                violations.append({
                    "validator": "bias",
                    "reason": f"Contains potentially biased phrasing: '{phrase}'",
                    "severity": "low",
                })
        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Sanitize text by redacting PII and annotating harmful sections."""
        sanitized = text

        for violation in violations:
            if violation.get("validator") == "pii":
                for match in violation.get("matches", []):
                    sanitized = sanitized.replace(match, "[REDACTED]")

        if any(v.get("validator") == "harmful_content" for v in violations):
            sanitized = "[Content trimmed for safety] " + sanitized

        return sanitized
