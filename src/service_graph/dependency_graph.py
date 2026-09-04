"""
RootIQ - Service Dependency Graph
Builds, validates, and queries directed service dependency graphs using NetworkX.
"""

import networkx as nx
import json
from typing import List, Tuple, Dict, Any, Optional

class ServiceDependencyGraph:
    def __init__(self, name: str = "RootIQ Microservice Topology"):
        self.name = name
        self.graph = nx.DiGraph()

    def add_service(self, service_name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add service node with optional metadata (e.g. tier, team, critical)."""
        attrs = attributes or {}
        self.graph.add_node(service_name.strip().lower(), **attrs)

    def add_dependency(self, caller: str, callee: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Add directed dependency edge: caller -> callee.
        (e.g., frontend calls api_gateway, api_gateway calls order_service).
        """
        caller_clean = caller.strip().lower()
        callee_clean = callee.strip().lower()
        self.add_service(caller_clean)
        self.add_service(callee_clean)
        attrs = attributes or {}
        self.graph.add_edge(caller_clean, callee_clean, **attrs)

    def load_from_edge_list(self, edges: List[Tuple[str, str]]) -> "ServiceDependencyGraph":
        """Load topology from list of (caller, callee) tuples."""
        for u, v in edges:
            self.add_dependency(u, v)
        return self

    def load_from_json(self, filepath: str) -> "ServiceDependencyGraph":
        """Load graph from JSON structure."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for node in data.get("nodes", []):
            if isinstance(node, dict):
                self.add_service(node["id"], node.get("attributes", {}))
            else:
                self.add_service(str(node))
        for edge in data.get("edges", []):
            self.add_dependency(edge["source"], edge["target"], edge.get("attributes", {}))
        return self

    def to_json(self) -> Dict[str, Any]:
        """Serialize graph to dictionary representation."""
        return {
            "name": self.name,
            "nodes": [{"id": n, "attributes": self.graph.nodes[n]} for n in self.graph.nodes],
            "edges": [{"source": u, "target": v, "attributes": self.graph.edges[u, v]} for u, v in self.graph.edges]
        }

    def get_services(self) -> List[str]:
        return list(self.graph.nodes())

    def get_downstream_dependents(self, service: str) -> List[str]:
        """Services that call this service or are downstream in failure cascade."""
        svc = service.strip().lower()
        if not self.graph.has_node(svc):
            return []
        # In call graph: A -> B means A calls B. If B fails, A is affected.
        # So callers of B are ancestors of B!
        return list(nx.ancestors(self.graph, svc))

    def get_dependencies_called(self, service: str) -> List[str]:
        """Services that this service calls (descendants in call graph)."""
        svc = service.strip().lower()
        if not self.graph.has_node(svc):
            return []
        return list(nx.descendants(self.graph, svc))