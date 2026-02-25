"""
AgentTorch-based population-scale lending market simulation.

Ports Loanville2's commercial lending economics into MIT's AgentTorch framework
for differentiable, GPU-accelerated simulation of 10K-1M+ borrower agents.

Architecture:
  - Substep 0: Loan Application (borrower demand)
  - Substep 1: Underwriting (lender credit decisions)
  - Substep 2: Adjudication (competitive auction)
  - Substep 3: Repayment (monthly amortization)
  - Substep 4: Defaults (differentiable default process)
  - Substep 5: Market Update (macro/cycle dynamics)
"""

try:
    from agent_torch.core.registry import Registry
except ImportError:
    # Provide a stub Registry so the module can be imported for inspection
    # even without agent-torch installed. Substep registration will be no-ops.
    class _StubRegistry:
        helpers = {
            "transition": {},
            "observation": {},
            "policy": {},
            "initialization": {},
            "network": {},
        }

        @classmethod
        def register_substep(cls, name, key):
            def decorator(klass):
                cls.helpers[key][name] = klass
                return klass
            return decorator

    Registry = _StubRegistry

from .substeps import *  # noqa: F401,F403 — triggers @Registry.register_substep

registry = Registry()
