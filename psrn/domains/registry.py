"""Domain registry for HierPose application domains.

Provides a central lookup for all registered domains, allowing
plugin-style extension via register_domain().

Usage:
    from psrn.domains.registry import get_domain, list_domains

    domain = get_domain("squat_form")
    print(domain.class_names)

    # Extend with a custom domain:
    register_domain("my_domain", MyCustomDomain)
"""

from __future__ import annotations

from typing import Dict, List, Type

from psrn.domains.base import BaseDomain
from psrn.domains.medical import RehabMonitoringDomain, XrayPositioningDomain
from psrn.domains.sports import GenericSportsDomain, SquatFormDomain
from psrn.domains.ergonomics import WorkplaceErgonomicsDomain


# ─────────────────────────────────────────────────────────────
# Registry: maps domain_name → class (not instance)
# ─────────────────────────────────────────────────────────────

DOMAIN_REGISTRY: Dict[str, Type[BaseDomain]] = {
    "xray_positioning":     XrayPositioningDomain,
    "rehab_monitoring":     RehabMonitoringDomain,
    "squat_form":           SquatFormDomain,
    "sports_generic":       GenericSportsDomain,
    "workplace_ergonomics": WorkplaceErgonomicsDomain,
}


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def get_domain_registry() -> Dict[str, BaseDomain]:
    """Return a dict of instantiated domain objects keyed by domain_name.

    Each domain is instantiated with default constructor arguments.
    Domains that require custom arguments (e.g. GenericSportsDomain)
    are instantiated with their defaults.

    Returns:
        Dict[str, BaseDomain] mapping domain_name → instance.
    """
    instances: Dict[str, BaseDomain] = {}
    for name, cls in DOMAIN_REGISTRY.items():
        try:
            instances[name] = cls()
        except TypeError:
            # Some domains may require positional args — skip gracefully
            pass
    return instances


def get_domain(name: str) -> BaseDomain:
    """Retrieve and instantiate a domain by name.

    Args:
        name: domain_name string (e.g. "squat_form", "workplace_ergonomics")

    Returns:
        Instantiated BaseDomain subclass.

    Raises:
        ValueError: if name is not found in the registry, listing available names.
    """
    cls = DOMAIN_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(DOMAIN_REGISTRY.keys()))
        raise ValueError(
            f"Domain '{name}' not found in registry. "
            f"Available domains: [{available}]"
        )
    return cls()


def list_domains() -> List[str]:
    """Return a sorted list of all registered domain names.

    Returns:
        List[str] of domain_name strings.
    """
    return sorted(DOMAIN_REGISTRY.keys())


def register_domain(name: str, cls: Type[BaseDomain]) -> None:
    """Register a custom domain class in the global registry.

    Allows plugin-style extension without modifying this file.
    If a domain with the same name already exists it will be overwritten.

    Args:
        name: unique domain_name string identifier
        cls:  BaseDomain subclass (not instance)

    Raises:
        TypeError: if cls is not a subclass of BaseDomain.

    Example:
        from psrn.domains.registry import register_domain
        from my_package import MyDomain
        register_domain("my_domain", MyDomain)
    """
    if not (isinstance(cls, type) and issubclass(cls, BaseDomain)):
        raise TypeError(
            f"cls must be a subclass of BaseDomain, got {cls!r}"
        )
    DOMAIN_REGISTRY[name] = cls
