"""Rule-based strategies replicating Demir (2023) Ch. 12 RB approach."""

from mibel_trading.strategies.rule_based import (
    BaseStrategy,
    NaiveBuyHoldStrategy,
    RandomStrategy,
    RuleBasedSpreadStrategy,
)

__all__ = [
    "BaseStrategy",
    "NaiveBuyHoldStrategy",
    "RandomStrategy",
    "RuleBasedSpreadStrategy",
]
