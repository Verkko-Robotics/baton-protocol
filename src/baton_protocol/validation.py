"""JSON Schema validation, as an optional extra.

The core is dependency-free, so ``jsonschema`` is not a hard requirement. This
module is the supported way to get real validation when you want it::

    pip install baton-protocol[validation]

Constructing :class:`JsonSchemaValidator` without the dependency raises rather
than degrading to a no-op. A validator that silently passes everything is worse
than having none, because the output contract looks enforced when it isn't.
"""

from __future__ import annotations

from typing import Any

from .interfaces import SchemaValidator


class JsonSchemaValidator(SchemaValidator):
    """Validator backed by ``jsonschema`` (Draft 2020-12)."""

    def __init__(self) -> None:
        try:
            import jsonschema  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "JsonSchemaValidator requires the 'validation' extra: "
                "pip install baton-protocol[validation]"
            ) from exc

    def validate(self, instance: Any, schema: dict[str, Any]) -> list[str]:
        """Return human-readable violations, most structural first.

        Messages are phrased for a language model to act on, since they are fed
        back into the loop as a correction prompt.
        """
        import jsonschema

        validator = jsonschema.Draft202012Validator(schema)
        return [
            f"{'/'.join(str(p) for p in error.path) or '<root>'}: {error.message}"
            for error in validator.iter_errors(instance)
        ]
