"""
RootIQ - Dependency Feature Engineering
Computes graph-topology features (in/out degree, PageRank, betweenness)
and upstream/downstream anomaly densities for root-cause inference.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional

class DependencyFeatureEngineer:
    def __init__(self, service_graph: Optional[nx.DiGraph] = None, service_col: str = "service"):
        self.service_graph = service_graph
        self.service_col = service_col

    def set_graph(self, graph: nx.DiGraph) -> None:
        self.service_graph = graph

    def compute_graph_structural_features(self) -> pd.DataFrame:
        """Calculate static topological features for each service in the graph."""
        if self.service_graph is None or len(self.service_graph.nodes) == 0:
            return pd.DataFrame(columns=[self.service_col, "in_degree", "out_degree", "pagerank", "betweenness"])

        g = self.service_graph
        nodes = list(g.nodes())
        
        in_degrees = dict(g.in_degree())
        out_degrees = dict(g.out_degree())
        pagerank = nx.pagerank(g) if len(nodes) > 1 else {n: 1.0 for n in nodes}
        betweenness = nx.betweenness_centrality(g) if len(nodes) > 1 else {n: 0.0 for n in nodes}

        records = []
        for n in nodes:
            records.append({
                self.service_col: n,
                "in_degree": in_degrees.get(n, 0),
                "out_degree": out_degrees.get(n, 0),
                "pagerank": round(pagerank.get(n, 0.0), 4),
                "betweenness": round(betweenness.get(n, 0.0), 4),
                "downstream_count": len(nx.descendants(g, n)) if g.has_node(n) else 0,
                "upstream_count": len(nx.ancestors(g, n)) if g.has_node(n) else 0
            })
        return pd.DataFrame(records)

    def attach_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge static graph metrics onto telemetry dataframe."""
        structural = self.compute_graph_structural_features()
        if structural.empty or self.service_col not in df.columns:
            return df
        return df.merge(structural, on=self.service_col, how="left").fillna(0.0)

    def compute_upstream_anomaly_density(self, df: pd.DataFrame, anomaly_col: str = "is_anomaly") -> pd.DataFrame:
        """Calculate the proportion of upstream services that are currently anomalous."""
        if self.service_graph is None or anomaly_col not in df.columns:
            return df
            
        df = df.copy()
        g = self.service_graph
        
        # For each time slice, compute upstream anomaly count
        upstream_rates = []
        for timestamp, t_group in df.groupby("timestamp"):
            service_status = dict(zip(t_group[self.service_col], t_group[anomaly_col]))
            for _, row in t_group.iterrows():
                svc = row[self.service_col]
                ancestors = list(nx.ancestors(g, svc)) if g.has_node(svc) else []
                if ancestors:
                    anom_count = sum(service_status.get(anc, 0) for anc in ancestors)
                    density = anom_count / len(ancestors)
                else:
                    anom_count = 0
                    density = 0.0
                upstream_rates.append({
                    "timestamp": timestamp,
                    self.service_col: svc,
                    "upstream_anomaly_count": anom_count,
                    "upstream_anomaly_density": round(density, 3)
                })

        up_df = pd.DataFrame(upstream_rates)
        return df.merge(up_df, on=["timestamp", self.service_col], how="left").fillna(0.0)