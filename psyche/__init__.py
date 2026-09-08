"""psyche-engine: Bayesian persona engine.

Conjugate-update mixture model over persona attributes with raking-based
design weights, recency decay, and drift-bounded tracking. See
docs/ALGORITHM.md for the math and SPEC.md for module interfaces.
"""

from psyche.models import BetaAttribute, DirichletAttribute, Persona, PersonaStore
from psyche.update import Confirmation

__version__ = "0.1.0"

__all__ = [
    "BetaAttribute",
    "Confirmation",
    "DirichletAttribute",
    "Persona",
    "PersonaStore",
    "__version__",
]
