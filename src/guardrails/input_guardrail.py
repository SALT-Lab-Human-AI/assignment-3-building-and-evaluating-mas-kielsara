"""
Input Guardrail
Checks user inputs for safety violations.
"""

from typing import Dict, Any, List, Optional
import re


class InputGuardrail:
    """
    Guardrail for checking input safety.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Define validation rules
    - Implement custom validators
    - Handle different types of violations
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.topic = config.get("system", {}).get("topic", "HCI research")
        self.prohibited_categories = config.get("safety", {}).get("prohibited_categories", [])

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query.

        Args:
            query: User input to validate

        Returns:
            Validation result

        TODO: YOUR CODE HERE
        - Implement validation logic
        - Check for toxic language
        - Check for prompt injection attempts
        - Check query length and format
        - Check for off-topic queries
        """
        violations = []

        # Length checks
        if len(query.strip()) < 5:
            violations.append({
                "validator": "length",
                "reason": "Query too short",
                "severity": "low",
            })

        if len(query) > 2000:
            violations.append({
                "validator": "length",
                "reason": "Query too long",
                "severity": "medium",
            })

        # Toxic or harmful language
        violations.extend(self._check_toxic_language(query))

        # Prompt injection attempts
        violations.extend(self._check_prompt_injection(query))

        # Off-topic queries (very lightweight relevance check to keep the system on UX/HCI)
        violations.extend(self._check_relevance(query))

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "sanitized_input": query.strip(),
        }

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """Very small, regex-based toxicity detector to avoid extra dependencies."""
        violations = []
        toxic_keywords = [
            "kill",
            "hurt",
            "attack",
            "exploit",
            "bomb",
            "harass",
            "self-harm",
            "suicide",
        ]

        lowered = text.lower()
        for keyword in toxic_keywords:
            if keyword in lowered:
                violations.append({
                    "validator": "toxicity",
                    "reason": f"Contains potentially harmful term: {keyword}",
                    "severity": "high",
                })

        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """Detect simple prompt injection phrases."""
        violations = []
        injection_patterns = [
            "ignore previous",
            "disregard",
            "forget all",
            "system:",
            "sudo",
            "execute shell",
            "bypass",
            "override",
        ]

        for pattern in injection_patterns:
            if pattern.lower() in text.lower():
                violations.append({
                    "validator": "prompt_injection",
                    "reason": f"Potential prompt injection: {pattern}",
                    "severity": "high",
                })

        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """Lightweight relevance check to keep queries on the configured topic."""
        violations = []

        # If prohibited categories are configured, block obvious mismatches
        prohibited = self.prohibited_categories or []
        lowered = query.lower()
        for category in prohibited:
            if category.replace("_", " ") in lowered:
                violations.append({
                    "validator": "prohibited_category",
                    "reason": f"Falls into prohibited category: {category}",
                    "severity": "high",
                })

        # Soft relevance check based on topic keywords
        topic_keywords = re.findall(r"[A-Za-z]+", self.topic.lower())
        if topic_keywords:
            matches = sum(1 for kw in topic_keywords if kw in lowered)
            # If no overlap, flag as potentially off-topic but low severity
            if matches == 0:
                violations.append({
                    "validator": "relevance",
                    "reason": f"Query may be off-topic relative to '{self.topic}'",
                    "severity": "low",
                })

        return violations
