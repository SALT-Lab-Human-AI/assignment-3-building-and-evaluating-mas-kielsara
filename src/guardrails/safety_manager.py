"""
Safety Manager
Coordinates safety guardrails and logs safety events.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
from pathlib import Path

from .input_guardrail import InputGuardrail
from .output_guardrail import OutputGuardrail


class SafetyManager:
    """
    Manages safety guardrails for the multi-agent system.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Define safety policies
    - Implement logging of safety events
    - Handle different violation types with appropriate responses
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety manager.

        Args:
            config: Safety configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.log_events = config.get("log_events", True)
        self.logger = logging.getLogger("safety")
        self.log_file = config.get("safety_log_file") or config.get("logging", {}).get("safety_log")

        # Safety event log
        self.safety_events: List[Dict[str, Any]] = []

        # Prohibited categories
        self.prohibited_categories = config.get("prohibited_categories", [
            "harmful_content",
            "personal_attacks",
            "misinformation",
            "off_topic_queries"
        ])

        # Violation response strategy
        self.on_violation = config.get("on_violation", {})
        self.input_guardrail = InputGuardrail({"system": config.get("system", {}), "safety": config})
        self.output_guardrail = OutputGuardrail({"system": config.get("system", {})})

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if input query is safe to process.

        Args:
            query: User query to check

        Returns:
            Dictionary with 'safe' boolean and optional 'violations' list

        TODO: YOUR CODE HERE
        - Implement guardrail checks
        - Detect harmful/inappropriate content
        - Detect off-topic queries
        - Return detailed violation information
        """
        if not self.enabled:
            return {"safe": True, "violations": [], "sanitized": query}

        result = self.input_guardrail.validate(query)
        is_safe = result.get("valid", True)
        violations = result.get("violations", [])

        if not is_safe and self.log_events:
            self._log_safety_event("input", query, violations, is_safe)

        response_action = self.on_violation.get("action", "refuse") if not is_safe else None
        response_message = None
        if not is_safe:
            if response_action == "sanitize":
                response_message = result.get("sanitized_input", "")
            else:
                response_message = self.on_violation.get(
                    "message",
                    "I cannot process this request due to safety policies."
                )

        return {
            "safe": is_safe,
            "violations": violations,
            "sanitized": result.get("sanitized_input", query),
            "action": response_action,
            "message": response_message,
        }

    def check_output_safety(self, response: str, sources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Check if output response is safe to return.

        Args:
            response: Generated response to check

        Returns:
            Dictionary with 'safe' boolean and optional 'violations' list

        TODO: YOUR CODE HERE
        - Implement output guardrail checks
        - Detect harmful content in responses
        - Detect potential misinformation
        - Sanitize or redact unsafe content
        """
        if not self.enabled:
            return {"safe": True, "response": response, "violations": []}

        guard_result = self.output_guardrail.validate(response, sources)
        is_safe = guard_result.get("valid", True)
        violations = guard_result.get("violations", [])
        sanitized_response = guard_result.get("sanitized_output", response)

        if not is_safe and self.log_events:
            self._log_safety_event("output", response, violations, is_safe)

        if not is_safe:
            action = self.on_violation.get("action", "refuse")
            if action == "refuse":
                sanitized_response = self.on_violation.get(
                    "message",
                    "I cannot provide this response due to safety policies."
                )
            # sanitize action already handled above

        return {
            "safe": is_safe,
            "violations": violations,
            "response": sanitized_response,
        }

    def _sanitize_response(self, response: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize response by removing or redacting unsafe content.

        TODO: YOUR CODE HERE Implement sanitization logic
        """
        # Placeholder
        return "[REDACTED] " + response

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool
    ):
        """
        Log a safety event.

        Args:
            event_type: "input" or "output"
            content: The content that was checked
            violations: List of violations found
            is_safe: Whether content passed safety checks
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "violations": violations,
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        }

        self.safety_events.append(event)
        self.logger.warning(f"Safety event: {event_type} - safe={is_safe}")

        # Write to safety log file if configured
        if self.log_file and self.log_events:
            try:
                Path(self.log_file).parent.mkdir(exist_ok=True, parents=True)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write safety log: {e}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Get all logged safety events."""
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Get statistics about safety events.

        Returns:
            Dictionary with safety statistics
        """
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])

        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0
        }

    def clear_events(self):
        """Clear safety event log."""
        self.safety_events = []
