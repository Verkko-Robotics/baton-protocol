"""Information-flow audit: which capability can read which fact, and how.

A specialist can learn a fact through three independent channels, each governed
by a different knob on a different object:

``envelope``
    ``SpecialistConfig.context_keys`` carves keys out of ``context_data``.
``blackboard``
    Another capability's published output, governed by that capability's
    ``output_visible_to`` (and, for entries already posted, each entry's
    ``visible_to``).
``delegation``
    A capability that may delegate to another can have the delegate do work
    with the *delegate's* slice and tools, and receive its output. Reach is
    transitive, so it is the channel most easily forgotten when reviewing grants.

Reviewing three knobs by eye does not scale past a handful of specialists, and a
leak is silent: nothing fails, a specialist just sees more than intended. This
module computes the answer from the same declarations the runtime enforces, so
it can be asserted in tests and printed in a review.

Pure and I/O-free: it reads an envelope and a registry and changes neither.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .blackboard import Blackboard
from .models import VISIBLE_TO_ALL, ContextEnvelope, SpecialistConfig

CHANNEL_ENVELOPE = "envelope"
CHANNEL_BLACKBOARD = "blackboard"
CHANNEL_DELEGATION = "delegation"


@dataclass(frozen=True)
class CapabilityFlow:
    """Everything one capability can learn, grouped by channel."""

    capability: str
    #: ``context_data`` keys the capability's own slice contains.
    envelope_keys: frozenset[str] = frozenset()
    #: ``context_keys`` entries that name nothing present in ``context_data``.
    #: Usually a typo, and the specialist silently gets less than intended.
    envelope_keys_missing: frozenset[str] = frozenset()
    #: Blackboard addresses (``capability.key``) it may read. Includes the
    #: ``<capability>.output`` of every registered capability that grants it
    #: access, plus entries already on the blackboard.
    blackboard_addresses: frozenset[str] = frozenset()
    #: Capabilities reachable by delegation, transitively.
    delegates: frozenset[str] = frozenset()
    #: Envelope keys reachable only through a delegate, mapped to the
    #: delegates that hold them. Excludes keys the capability already has.
    delegated_envelope_keys: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def channels_for_envelope_key(self, key: str) -> frozenset[str]:
        channels: set[str] = set()
        if key in self.envelope_keys:
            channels.add(CHANNEL_ENVELOPE)
        if key in self.delegated_envelope_keys:
            channels.add(CHANNEL_DELEGATION)
        return frozenset(channels)


@dataclass(frozen=True)
class InformationFlowReport:
    """Per-capability flows for one envelope and registry."""

    flows: Mapping[str, CapabilityFlow]

    def __getitem__(self, capability: str) -> CapabilityFlow:
        return self.flows[capability]

    def __contains__(self, capability: object) -> bool:
        return capability in self.flows

    def readers_of_envelope_key(self, key: str) -> dict[str, frozenset[str]]:
        """Capabilities that can learn ``context_data[key]``, with channels."""
        out: dict[str, frozenset[str]] = {}
        for capability, flow in self.flows.items():
            channels = flow.channels_for_envelope_key(key)
            if channels:
                out[capability] = channels
        return out

    def readers_of_address(self, address: str) -> frozenset[str]:
        """Capabilities that can read blackboard ``address`` directly."""
        return frozenset(
            capability
            for capability, flow in self.flows.items()
            if address in flow.blackboard_addresses
        )

    def to_dict(self) -> dict[str, dict]:
        """JSON-friendly form for logs and review artifacts."""
        return {
            capability: {
                "envelope_keys": sorted(flow.envelope_keys),
                "envelope_keys_missing": sorted(flow.envelope_keys_missing),
                "blackboard_addresses": sorted(flow.blackboard_addresses),
                "delegates": sorted(flow.delegates),
                "delegated_envelope_keys": {
                    key: sorted(via)
                    for key, via in sorted(flow.delegated_envelope_keys.items())
                },
            }
            for capability, flow in sorted(self.flows.items())
        }


def audit_information_flow(
    envelope: ContextEnvelope,
    registry: Mapping[str, SpecialistConfig],
) -> InformationFlowReport:
    """Compute what every registered capability can read, and through which channel.

    Delegation targets that are not in ``registry`` are ignored: the runtime
    refuses them with ``CapabilityNotRegistered``, so they confer nothing.
    """
    context_data = envelope.context_data or {}
    existing = Blackboard(
        envelope.task_id,
        envelope.tenant_id,
        entries=envelope.blackboard_snapshot,
    )

    flows: dict[str, CapabilityFlow] = {}
    for capability, config in registry.items():
        own_keys = frozenset(k for k in config.context_keys if k in context_data)
        missing = frozenset(k for k in config.context_keys if k not in context_data)

        addresses = set(existing.read_all(capability))
        for producer, producer_config in registry.items():
            if producer == capability:
                continue  # a capability always reads its own output
            if _output_readable_by(producer_config, capability):
                addresses.add(f"{producer}.output")
        # Its own output is always readable by itself.
        addresses.add(f"{capability}.output")

        delegates = _reachable_delegates(capability, registry)
        delegated: dict[str, set[str]] = {}
        for delegate in delegates:
            for key in registry[delegate].context_keys:
                if key in context_data and key not in own_keys:
                    delegated.setdefault(key, set()).add(delegate)

        flows[capability] = CapabilityFlow(
            capability=capability,
            envelope_keys=own_keys,
            envelope_keys_missing=missing,
            blackboard_addresses=frozenset(addresses),
            delegates=frozenset(delegates),
            delegated_envelope_keys={k: frozenset(v) for k, v in delegated.items()},
        )
    return InformationFlowReport(flows=flows)


def _output_readable_by(producer: SpecialistConfig, reader: str) -> bool:
    """Whether ``producer``'s published output is visible to ``reader``.

    Mirrors ``Orchestrator._publish`` (``visible_to=config.output_visible_to``)
    and ``Blackboard.post`` (``None`` -> everyone, empty -> nobody).
    """
    visible = producer.output_visible_to
    if visible is None:
        return True
    return VISIBLE_TO_ALL in visible or reader in visible


def _reachable_delegates(
    start: str, registry: Mapping[str, SpecialistConfig]
) -> set[str]:
    """Capabilities ``start`` can reach by delegating, excluding itself."""
    seen: set[str] = set()
    stack = list(registry[start].can_delegate_to)
    while stack:
        name = stack.pop()
        if name in seen or name not in registry:
            continue
        seen.add(name)
        stack.extend(registry[name].can_delegate_to)
    seen.discard(start)
    return seen
