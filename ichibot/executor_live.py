"""Live order executor -- deliberately not implemented yet.

This file exists to keep live-order code physically separated from the dry-run
path. Real order placement, transaction signing, and live-trading guardrails
are added in a later milestone (M8), and only after dry-run paper trading has been
verified.

Until then, constructing LiveExecutor raises. There is intentionally no code in
this file that is capable of placing an order.
"""

from __future__ import annotations


class LiveExecutor:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Live trading is not implemented yet. The live executor is added in a "
            "later milestone with full guardrails (key handling, 1x verification, "
            "explicit confirmation). Use the dry-run executor until then."
        )
