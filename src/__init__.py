"""Package root for the KLA PS01 restoration project.

Kept import-light on purpose: inference.py imports from src and its import cost sits
inside KLA's measured wall-clock window (SPEC 11.2, docs/decisions.md D7).
Do NOT add eager submodule imports here.
"""

__all__: list[str] = []
