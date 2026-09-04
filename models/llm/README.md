# Local LLM Integration Guide for RootIQ

RootIQ is designed with a **privacy-first, local-first architecture**. Paid external APIs (like OpenAI) are NOT required.

## Recommended Model
- **Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Parameters**: 3 Billion
- **License**: Apache 2.0 (Open-Source)
- **Minimum RAM**: 8 GB (CPU) or 4 GB VRAM (GPU)

## How It Works in RootIQ
1. **Decision Support, Not Decision Maker**: The analytical Root Cause Engine (Isolation Forest + Time-Series Onset + NetworkX Graph) computes the rankings and evidence first.
2. The LLM converts this structured multi-dimensional evidence into natural-language incident reports for engineers.
3. **Automatic Fallback**: If you do not install torch/transformers or your laptop has low RAM, RootIQ seamlessly uses the built-in deterministic rule-based explainer.

## How to Enable Local LLM (Optional)
```bash
pip install torch transformers accelerate
```
In your code or Streamlit dashboard, set `use_local_llm=True` in the sidebar!