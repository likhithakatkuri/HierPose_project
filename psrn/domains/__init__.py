"""HierPose domain modules.

Each domain defines the context in which pose classification is applied:
- pose class definitions
- reference poses (target/ideal poses)
- feedback templates (domain language)
- severity thresholds for counterfactual guidance

Available domains:
    medical    — XrayPositioningDomain, RehabMonitoringDomain
    sports     — SquatFormDomain, GenericSportsDomain
    ergonomics — WorkplaceErgonomicsDomain
    registry   — get_domain_registry(), list_domains()
"""

from psrn.domains.registry import get_domain_registry, list_domains, get_domain

__all__ = ["get_domain_registry", "list_domains", "get_domain"]
