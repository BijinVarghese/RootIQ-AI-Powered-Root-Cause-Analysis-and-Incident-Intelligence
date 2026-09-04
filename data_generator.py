"""
RootIQ - Synthetic Telemetry & Benchmark Incident Generator
Generates realistic multi-service logs, metrics, traces, and ground-truth incidents
simulating microservices cascading failure patterns.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_telemetry(output_dir: str = "data", seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)

    services = [
        "frontend",
        "api_gateway",
        "order_service",
        "inventory_service",
        "payment_service",
        "database",
        "cache"
    ]

    # Call graph dependencies (caller -> callee)
    dependencies = [
        ("frontend", "api_gateway"),
        ("api_gateway", "order_service"),
        ("api_gateway", "inventory_service"),
        ("order_service", "payment_service"),
        ("order_service", "database"),
        ("inventory_service", "database"),
        ("inventory_service", "cache"),
        ("payment_service", "database")
    ]

    # Save service topology JSON
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw", "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw", "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw", "traces"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "evaluation", "labelled_incidents"), exist_ok=True)

    topology_path = os.path.join(output_dir, "raw", "service_dependencies.json")
    with open(topology_path, "w", encoding="utf-8") as f:
        json.dump({
            "nodes": services,
            "edges": [{"source": u, "target": v} for u, v in dependencies]
        }, f, indent=4)

    # 120 minutes of telemetry at 1-minute steps
    start_time = datetime(2026, 3, 1, 10, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(120)]

    metrics_rows = []
    logs_rows = []
    traces_rows = []

    # Incident definitions
    # Incident 1: Database Connection Pool Exhaustion (Minutes 40 - 55)
    # Root Cause: database
    inc1_start, inc1_end = 40, 55

    # Incident 2: Payment Service Outage (Minutes 80 - 92)
    # Root Cause: payment_service
    inc2_start, inc2_end = 80, 92

    for t_idx, ts in enumerate(timestamps):
        iso_ts = ts.isoformat() + "Z"

        for svc in services:
            # Baseline normal parameters
            cpu = np.random.uniform(20.0, 45.0)
            mem = np.random.uniform(35.0, 55.0)
            lat = np.random.uniform(25.0, 75.0)
            err_rate = np.random.uniform(0.00, 0.01)
            req_rate = np.random.uniform(150.0, 300.0)
            log_level = "INFO"
            log_msg = f"Handled {int(req_rate)} requests successfully."
            trace_duration = lat + np.random.uniform(-5.0, 5.0)
            trace_status = 200

            # --- INCIDENT 1 INJECTION (Root Cause: database) ---
            if inc1_start <= t_idx <= inc1_end:
                if svc == "database":
                    # Earliest onset: Minute 40
                    cpu = np.random.uniform(85.0, 98.0)
                    mem = np.random.uniform(80.0, 92.0)
                    lat = np.random.uniform(1200.0, 2200.0)
                    err_rate = np.random.uniform(0.15, 0.40)
                    log_level = "CRITICAL"
                    log_msg = "Connection pool exhausted! Active connections: 200/200. Lock timeout on table 'orders'."
                    trace_duration = lat
                    trace_status = 500
                elif svc in ["order_service", "inventory_service"] and t_idx >= inc1_start + 2:
                    # Cascade onset: Minute 42
                    lat = np.random.uniform(1300.0, 2400.0)
                    err_rate = np.random.uniform(0.20, 0.50)
                    log_level = "ERROR"
                    log_msg = f"Database query timed out after 1000ms: Call to database failed."
                    trace_duration = lat
                    trace_status = 504
                elif svc in ["api_gateway", "frontend"] and t_idx >= inc1_start + 3:
                    # Downstream cascade: Minute 43
                    lat = np.random.uniform(1400.0, 2500.0)
                    err_rate = np.random.uniform(0.30, 0.65)
                    log_level = "ERROR"
                    log_msg = "Upstream gateway returned HTTP 504: Transaction processing failure."
                    trace_duration = lat
                    trace_status = 504

            # --- INCIDENT 2 INJECTION (Root Cause: payment_service) ---
            elif inc2_start <= t_idx <= inc2_end:
                if svc == "payment_service":
                    # Earliest onset: Minute 80
                    cpu = np.random.uniform(70.0, 88.0)
                    lat = np.random.uniform(800.0, 1500.0)
                    err_rate = np.random.uniform(0.50, 0.85)
                    log_level = "FATAL"
                    log_msg = "Payment Gateway SSL handshake failed: External payment provider unreachable."
                    trace_duration = lat
                    trace_status = 502
                elif svc == "order_service" and t_idx >= inc2_start + 1:
                    # Cascade onset: Minute 81
                    err_rate = np.random.uniform(0.40, 0.70)
                    lat = np.random.uniform(850.0, 1600.0)
                    log_level = "ERROR"
                    log_msg = "Checkout failed: Payment service returned HTTP 502 Bad Gateway."
                    trace_duration = lat
                    trace_status = 502
                elif svc in ["api_gateway", "frontend"] and t_idx >= inc2_start + 2:
                    # Cascade onset: Minute 82
                    err_rate = np.random.uniform(0.25, 0.45)
                    log_level = "WARN"
                    log_msg = "Order placement returned partial failure."
                    trace_status = 500

            # Append Records
            metrics_rows.append({
                "timestamp": iso_ts,
                "service": svc,
                "cpu_usage": round(cpu, 2),
                "memory_usage": round(mem, 2),
                "latency_ms": round(lat, 2),
                "request_rate": round(req_rate, 2),
                "error_rate": round(err_rate, 4)
            })

            logs_rows.append({
                "timestamp": iso_ts,
                "service": svc,
                "level": log_level,
                "message": log_msg
            })

            traces_rows.append({
                "timestamp": iso_ts,
                "trace_id": f"trc-{t_idx:04d}-{svc[:3]}",
                "service": svc,
                "duration_ms": round(trace_duration, 2),
                "status_code": trace_status
            })

    # Save CSVs
    pd.DataFrame(metrics_rows).to_csv(os.path.join(output_dir, "raw", "metrics", "telemetry_metrics.csv"), index=False)
    pd.DataFrame(logs_rows).to_csv(os.path.join(output_dir, "raw", "logs", "telemetry_logs.csv"), index=False)
    pd.DataFrame(traces_rows).to_csv(os.path.join(output_dir, "raw", "traces", "telemetry_traces.csv"), index=False)

    # Save Ground Truth Incident Benchmark
    ground_truth = [
        {
            "incident_id": "INC-20260301-001",
            "start_time": timestamps[inc1_start].isoformat() + "Z",
            "end_time": timestamps[inc1_end].isoformat() + "Z",
            "true_root_cause": "database",
            "scenario": "Database Connection Pool Exhaustion and Lock Spike",
            "affected_services": ["database", "order_service", "inventory_service", "api_gateway", "frontend"],
            "propagation_chain": ["database", "order_service", "api_gateway", "frontend"]
        },
        {
            "incident_id": "INC-20260301-002",
            "start_time": timestamps[inc2_start].isoformat() + "Z",
            "end_time": timestamps[inc2_end].isoformat() + "Z",
            "true_root_cause": "payment_service",
            "scenario": "Third-Party Payment Gateway Outage",
            "affected_services": ["payment_service", "order_service", "api_gateway", "frontend"],
            "propagation_chain": ["payment_service", "order_service", "api_gateway", "frontend"]
        }
    ]

    with open(os.path.join(output_dir, "evaluation", "labelled_incidents", "ground_truth_incidents.json"), "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=4)

    print(f"Generated {len(metrics_rows)} metrics, {len(logs_rows)} logs, {len(traces_rows)} traces.")
    print("Telemetry dataset and ground-truth benchmark created successfully!")

if __name__ == "__main__":
    generate_telemetry("data")