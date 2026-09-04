"""
RootIQ - Service Graph Analyzer
Calculates topology metrics, propagation paths, and dependency evidence
for root-cause candidate evaluation.
"""

import networkx as nx
from typing import Dict, Any, List, Optional
from .dependency_graph import ServiceDependencyGraph

class ServiceGraphAnalyzer:
    def __init__(self, service_graph: ServiceDependencyGraph):
        self.sdg = service_graph
        self.g = service_graph.graph

    def compute_centrality(self) -> Dict[str, Dict[str, float]]:
        """Compute degree, betweenness, and in/out centrality per service."""
        nodes = list(self.g.nodes())
        if not nodes:
            return {}

        in_deg = dict(self.g.in_degree())
        out_deg = dict(self.g.out_degree())
        betweenness = nx.betweenness_centrality(self.g) if len(nodes) > 1 else {n: 0.0 for n in nodes}

        results = {}
        for n in nodes:
            results[n] = {
                "in_degree": in_deg.get(n, 0),
                "out_degree": out_deg.get(n, 0),
                "betweenness": round(betweenness.get(n, 0.0), 4)
            }
        return results

    def find_failure_propagation_path(self, root_candidate: str, affected_services: List[str]) -> List[List[str]]:
        """
        Find topological paths tracing how failure propagated from root candidate
        to other affected services.
        """
        root = root_candidate.strip().lower()
        affected = [s.strip().lower() for s in affected_services if s.strip().lower() != root]
        
        if not self.g.has_node(root) or not affected:
            return []

        paths = []
        for target in affected:
            # Check path where target calls root (callee failed causing caller error)
            if self.g.has_node(target):
                try:
                    # In call graph: caller -> callee.
                    # Failure flows backward: callee error propagates back to caller!
                    if nx.has_path(self.g, target, root):
                        p = nx.shortest_path(self.g, target, root)
                        paths.append(list(reversed(p))) # show from root -> ... -> target
                except nx.NetworkXNoPath:
                    pass
        return paths

    def calculate_dependency_score(self, candidate: str, affected_services: List[str]) -> float:
        """
        Calculates dependency evidence score in [0.0, 1.0].
        If candidate explains a high fraction of the affected services (they depend on it),
        the score is high.
        """
        cand = candidate.strip().lower()
        if not self.g.has_node(cand):
            return 0.2 # small baseline

        other_affected = [s.strip().lower() for s in affected_services if s.strip().lower() != cand]
        if not other_affected:
            return 0.5 # single service incident

        # Affected services that call the candidate (callee failure cascade)
        callers = set(nx.ancestors(self.g, cand))
        explained_count = sum(1 for s in other_affected if s in callers)
        score = explained_count / len(other_affected)
        
        # Give bonus if candidate is a deep backend service (high in-degree / database / cache)
        in_degree = self.g.in_degree(cand)
        bonus = min(0.2, in_degree * 0.05)

        return round(min(1.0, score + bonus), 3)