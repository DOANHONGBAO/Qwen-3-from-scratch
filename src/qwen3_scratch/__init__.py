"""qwen3_scratch: an educational, test-driven implementation of the Qwen3 dense Transformer.

Milestone status (see docs/roadmap.md):

* M0 (foundation): configuration schema and environment checks only.
* Model components (RMSNorm, RoPE, GQA, ...) are added in M1, one module at a time.

The package is laid out so that each Python module maps onto a future C++20
module in ``cpp/`` (see docs/architecture.md).
"""

from qwen3_scratch.config import Qwen3Config

__version__ = "0.0.1"

__all__ = ["Qwen3Config", "__version__"]
