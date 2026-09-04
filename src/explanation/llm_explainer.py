"""
RootIQ - Incident Explainer
Generates readable incident diagnosis summaries and mitigation recommendations.
Supports optional local Qwen2.5 3B Instruct via Hugging Face Transformers
with a high-fidelity deterministic analytical template fallback.
"""

import json
from typing import Dict, Any, List, Optional

class IncidentExplainer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", use_local_llm: bool = False):
        self.model_name = model_name
        self.use_local_llm = use_local_llm
        self.pipeline = None
        self._llm_loaded = False

    def load_llm(self) -> bool:
        """Attempt to load local Hugging Face model if requested."""
        if not self.use_local_llm:
            return False
        try:
            import torch
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                model_kwargs={"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            self._llm_loaded = True
            return True
        except Exception as e:
            print(f"[IncidentExplainer] Local LLM could not be loaded ({e}). Falling back to rule-based explainer.")
            self._llm_loaded = False
            return False

    def explain(self, incident_data: Dict[str, Any], ranked_causes: List[Dict[str, Any]]) -> str:
        """Generate human-interpretable incident explanation."""
        if self._llm_loaded and self.pipeline is not None:
            try:
                return self._generate_llm_explanation(incident_data, ranked_causes)
            except Exception as e:
                print(f"[IncidentExplainer] LLM inference error: {e}. Using rule-based fallback.")

        return self._generate_rule_based_explanation(incident_data, ranked_causes)

    def _generate_rule_based_explanation(self, incident: Dict[str, Any], ranked: List[Dict[str, Any]]) -> str:
        """Deterministic analytical incident report."""
        if not ranked:
            return "No anomalous services identified for root cause analysis."

        top1 = ranked[0]
        top_svc = top1["service"]
        top_score = top1["root_cause_score"]
        top_conf = top1["confidence_pct"]
        inc_id = incident.get("incident_id", "N/A")
        duration = incident.get("duration_minutes", 0)
        affected = ", ".join(incident.get("affected_services", []))

        # Propagation path text
        paths = top1.get("propagation_paths", [])
        if paths:
            path_str = " -> ".join(paths[0])
            cascade_text = f"The failure originated at **{top_svc}** and propagated downstream along the path: `{path_str}`."
        else:
            cascade_text = f"The failure originated at **{top_svc}**, which acts as an upstream dependency for other affected services."

        evidence_bullets = []
        evidence_bullets.append(f"- **Earliest Anomaly Onset**: `{top1['onset_time']}` (Temporal Precedence Score: {top1['temporal_evidence']*100:.0f}%)")
        evidence_bullets.append(f"- **Anomaly Magnitude**: Peak severity score {top1['anomaly_evidence']*100:.0f}% based on Isolation Forest telemetry deviation")
        
        details = top1.get("evidence_details", [])
        if details:
            evidence_bullets.append(f"- **Direct Metric Symptoms**: {', '.join(details)}")
        else:
            evidence_bullets.append(f"- **Direct Metric Symptoms**: Extreme latency jump and elevated error rate during the incident window")
            
        evidence_bullets.append(f"- **Dependency Impact**: Dependency graph score {top1['dependency_evidence']*100:.0f}% indicating callers were degraded by this component's failure")

        ev_text = "\n".join(evidence_bullets)

        alt_candidates = ""
        if len(ranked) > 1:
            alt_list = [f"{r['service']} ({r['root_cause_score']:.1f}/100, {r['confidence_pct']}%)" for r in ranked[1:3]]
            alt_candidates = f"\n**Secondary Downstream Candidates**:\n- " + "\n- ".join(alt_list)

        explanation = f"""### Incident Intelligence Summary: `{inc_id}`

**Primary Root Cause Identified:** `{top_svc}` (Confidence: {top_conf}%, Score: {top_score:.1f}/100)
**Incident Duration:** ~{duration} minutes | **Affected Services:** {affected}

#### Analytical Evidence:
{ev_text}

#### Failure Propagation & Impact:
{cascade_text}
{alt_candidates}

#### Recommended Remediation:
1. Inspect health logs, thread pools, and resource utilization on `{top_svc}` around `{top1['onset_time']}`.
2. Check database connection pool limits, slow queries, or third-party downstream timeouts.
3. Validate circuit breakers on caller services to prevent cascading failures across the dependency graph.
"""
        return explanation.strip()

    def _generate_llm_explanation(self, incident: Dict[str, Any], ranked: List[Dict[str, Any]]) -> str:
        """Prompt-engineered LLM inference."""
        prompt = f"""You are RootIQ, an expert SRE and AIOps decision-support AI.
Summarize the following incident analysis clearly and concisely for an engineering team:

Incident ID: {incident.get('incident_id')}
Affected Services: {incident.get('affected_services')}
Ranked Candidates: {json.dumps(ranked[:3], indent=2)}

Provide:
1. Executive Root Cause Verdict
2. Key Telemetry & Graph Evidence
3. Failure Propagation Chain
4. Actionable Remediation Steps
"""
        messages = [
            {"role": "system", "content": "You are a professional Site Reliability Engineering AI assistant."},
            {"role": "user", "content": prompt}
        ]
        text_prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(text_prompt, max_new_tokens=400, do_sample=False)
        return outputs[0]["generated_text"]