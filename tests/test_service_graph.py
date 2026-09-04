"""Unit tests for Service Dependency Graph."""
import pytest
from src.service_graph.dependency_graph import ServiceDependencyGraph
from src.service_graph.graph_analysis import ServiceGraphAnalyzer

def test_service_dependency_graph():
    sdg = ServiceDependencyGraph()
    sdg.add_dependency("frontend", "gateway")
    sdg.add_dependency("gateway", "order_service")
    sdg.add_dependency("order_service", "database")

    assert "database" in sdg.get_services()
    assert "order_service" in sdg.get_downstream_dependents("database")

    analyzer = ServiceGraphAnalyzer(sdg)
    centrality = analyzer.compute_centrality()
    assert "gateway" in centrality

    # If database fails and frontend, gateway, order are affected:
    # Failure propagation path exists
    paths = analyzer.find_failure_propagation_path("database", ["order_service", "gateway"])
    assert len(paths) > 0
    assert paths[0][0] == "database"