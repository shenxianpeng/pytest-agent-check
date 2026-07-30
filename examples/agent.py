"""
Example: a minimal support-agent used in the demo test below.

In real usage this would be your LangChain / OpenAI / custom agent.
Here we just simulate tool calls and LLM responses so the example
works offline without any API key.
"""

from __future__ import annotations

from typing import Any


class SupportAgent:
    """
    A mock support agent that can process simple customer requests.

    Real agents would connect to an LLM and external tools; this
    minimal version demonstrates the shape of the data that
    ``pytest-agent-eval`` expects.
    """

    def __init__(self, name: str = "support-agent") -> None:
        self.name = name

    def run(self, user_input: str) -> dict[str, Any]:
        """
        Process a user message and return tool calls + text output.

        Returns:
            A dict with:
            - ``tool_calls``: list of ``{"name", "arguments", "result"}``
            - ``output``: the final natural-language response.

        """
        text = user_input.lower()

        if "refund" in text:
            return self._handle_refund(text)
        if "balance" in text or "account" in text:
            return self._handle_balance(text)
        if "order" in text and ("status" in text or "where" in text):
            return self._handle_order_status(text)
        return {
            "tool_calls": [],
            "output": "I'm sorry, I couldn't understand your request. "
            "Please try rephrasing.",
        }

    # ── internal helpers ──────────────────────────────────────

    def _handle_refund(self, text: str) -> dict[str, Any]:
        # Parse order number if provided
        order_id = "ORD-001"
        for word in text.split():
            if word.startswith("ord-"):
                order_id = word.upper()
                break

        return {
            "tool_calls": [
                {
                    "name": "lookup_order",
                    "arguments": {"order_id": order_id},
                    "result": {
                        "order_id": order_id,
                        "status": "delivered",
                        "amount": 49.99,
                        "items": ["Wireless Headphones"],
                    },
                },
                {
                    "name": "check_refund_eligibility",
                    "arguments": {"order_id": order_id},
                    "result": {
                        "eligible": True,
                        "reason": "Within 30-day return window",
                        "refund_amount": 49.99,
                    },
                },
                {
                    "name": "issue_refund",
                    "arguments": {
                        "order_id": order_id,
                        "amount": 49.99,
                        "method": "original_payment",
                    },
                    "result": {
                        "refund_id": "RF-" + order_id[-3:],
                        "status": "approved",
                        "expected_arrival": "3-5 business days",
                    },
                },
            ],
            "output": (
                f"Your refund of $49.99 for order {order_id} has been approved "
                f"(refund ID: RF-{order_id[-3:]}). "
                f"The amount will be returned to your original payment method "
                f"within 3-5 business days."
            ),
        }

    def _handle_balance(self, text: str) -> dict[str, Any]:
        return {
            "tool_calls": [
                {
                    "name": "get_account_balance",
                    "arguments": {"account_id": "ACT-001"},
                    "result": {"balance": 1250.00, "currency": "USD"},
                }
            ],
            "output": "Your current account balance is $1,250.00 USD.",
        }

    def _handle_order_status(self, text: str) -> dict[str, Any]:
        return {
            "tool_calls": [
                {
                    "name": "lookup_order",
                    "arguments": {"order_id": "ORD-001"},
                    "result": {
                        "order_id": "ORD-001",
                        "status": "in_transit",
                        "estimated_delivery": "2025-02-20",
                        "carrier": "FastShip",
                        "tracking": "FS-987654321",
                    },
                }
            ],
            "output": (
                "Your order ORD-001 is currently in transit via FastShip "
                "(tracking: FS-987654321). Estimated delivery is February 20, 2025."
            ),
        }
