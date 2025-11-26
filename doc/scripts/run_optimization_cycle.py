import os
import json
import yaml
from datetime import datetime, timezone

import requests
from openai import OpenAI

# --------------------------------------------------------------------
# Environment configuration
# --------------------------------------------------------------------

NOTION_TOKEN = os.environ["NOTION_TOKEN"].strip()
NOTION_DB = os.environ["NOTION_LMO_DB"].strip()

# OPENAI_API_KEY is read automatically by the OpenAI client
client = OpenAI()

# --------------------------------------------------------------------
# LLM configuration
# --------------------------------------------------------------------
"""
We treat each entry here as a separate 'LLM' for scoring.

Right now they are all OpenAI models, but the structure is designed so you
can later plug in Anthropic / Gemini etc. by adding entries like:

{"name": "claude-3.5-sonnet", "backend": "anthropic", "model": "claude-3.5-sonnet"}
{"name": "gemini-1.5-flash", "backend": "gemini", "model": "gemini-1.5-flash"}

and extending `call_llm` below.
"""

LLM_CONFIGS = [
    {"name": "chatgpt-4.1-mini", "backend": "openai", "model": "gpt-4.1-mini"},
    {"name": "chatgpt-4.1", "backend": "openai", "model": "gpt-4.1"},
]


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def load_clients():
    """
    Load clients from doc/clients/clients.yaml
    """
    path = "doc/clients/clients.yaml"
    print(f"[LMO] Loading clients from: {os.path.abspath(path)}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "clients" not in data or not isinstance(data["clients"], list):
        raise ValueError(f"Unexpected clients.yaml structure: {data}")

    print(f"[LMO] Loaded {len(data['clients'])} client(s)")
    return data["clients"]


def notion_update(page_id: str, scores: dict):
    """
    Update a client's row in the LMO Client Tracker database.

    Properties used:
      - Visibility Score (number)
      - Accuracy Score   (number)
      - Drift Score      (number)
      - Automation Event (rich text)
      - Automation Timestamp (date)
      - Last Optimization Date (date)
      - LLM Breakdown (rich text)  <-- auto-created if not present
    """
    url = f"https://api.notion.com/v1/pages/{page_id}"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

    now_iso = datetime.now(timezone.utc).isoformat()

    body = {
        "properties": {
            "Visibility Score": {"number": scores["visibility"]},
            "Accuracy Score": {"number": scores["accuracy"]},
            "Drift Score": {"number": scores["drift"]},
            "Automation Event": {
                "rich_text": [
                    {"text": {"content": scores.get("event_label", "Weekly optimization cycle (multi-LLM)")}}
                ]
            },
            "Automation Timestamp": {"date": {"start": now_iso}},
            "Last Optimization Date": {"date": {"start": now_iso}},
            "LLM Breakdown": {
                "rich_text": [{"text": {"content": scores.get("breakdown", "")[:1900]}}],
            },
        }
    }

    print(f"[LMO] Updating Notion page {page_id} with overall scores: {scores}")
    resp = requests.patch(url, headers=headers, json=body)
    print(f"[LMO] Notion update status: {resp.status_code}")
    if not resp.ok:
        print(resp.text)
        resp.raise_for_status()


def build_prompt(client_obj: dict) -> str:
    """
    Build the evaluation prompt for a single client.
    """
    name = client_obj.get("name", "")
    industry = client_obj.get("industry", "")
    location = client_obj.get("location", "")
    canonical_facts = client_obj.get("canonical_facts", "")

    return f"""
You are an evaluator for Language Model Optimization (LMO).

Your job is to estimate how well large language models currently represent this company.

Company:
- Name: {name}
- Industry: {industry}
- Location: {location}

Canonical facts about the company (what is actually true):
{canonical_facts}

You must output STRICT JSON with this exact shape:

{{
  "visibility": 0.0-1.0,  // how likely LLMs are to mention this company when asked about its niche
  "accuracy": 0.0-1.0,    // how accurate the typical LLM answer is likely to be about this company
  "drift": 0.0-1.0,       // how likely it is that current LLM answers are stale, outdated, or misaligned
  "notes": "short explanation of why you chose these numbers"
}}

Rules:
- Be conservative rather than optimistic.
- Base your judgment only on the information provided plus general world knowledge.
- Respond ONLY with JSON. Do not include any commentary or formatting outside the JSON object.
    """.strip()


# --------------------------------------------------------------------
# LLM calling + aggregation
# --------------------------------------------------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def call_llm(config: dict, prompt: str) -> dict:
    """
    Call a single LLM config and return scores dict.

    For now only 'openai' backend is implemented.
    """
    backend = config["backend"]
    model = config["model"]
    name = config["name"]

    print(f"[LMO] Scoring with {name} ({backend}:{model})")

    if backend == "openai":
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an LMO scoring engine that returns strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content.strip()
            print(f"[LMO] Raw response from {name}: {content}")
            data = json.loads(content)

            visibility = clamp01(float(data.get("visibility", 0.0)))
            accuracy = clamp01(float(data.get("accuracy", 0.0)))
            drift = clamp01(float(data.get("drift", 0.0)))
            notes = str(data.get("notes", "")).strip()

            return {
                "visibility": visibility,
                "accuracy": accuracy,
                "drift": drift,
                "notes": notes,
            }

        except Exception as e:
            print(f"[LMO] Error calling {name}: {e}")
            return {
                "visibility": 0.0,
                "accuracy": 0.0,
                "drift": 0.0,
                "notes": f"Fallback scores for {name} due to error: {e}",
            }

    # Placeholder for future backends
    # elif backend == "anthropic":
    #     ...
    # elif backend == "gemini":
    #     ...

    else:
        print(f"[LMO] Unsupported backend: {backend}")
        return {
            "visibility": 0.0,
            "accuracy": 0.0,
            "drift": 0.0,
            "notes": f"Unsupported backend {backend}",
        }


def score_with_all_llms(prompt: str) -> dict:
    """
    Run scoring across all configured LLMs and aggregate results.
    Returns a dict with:
      - visibility / accuracy / drift (overall average)
      - breakdown: string summary
      - event_label: text for Automation Event
    """
    per_llm = {}
    vis_list, acc_list, drift_list = [], [], []
    notes_chunks = []

    for cfg in LLM_CONFIGS:
        name = cfg["name"]
        scores = call_llm(cfg, prompt)
        per_llm[name] = scores

        vis_list.append(scores["visibility"])
        acc_list.append(scores["accuracy"])
        drift_list.append(scores["drift"])

        notes_chunks.append(f"{name}: {scores['notes']}")

    if not vis_list:
        # Should not happen, but be safe
        overall_visibility = overall_accuracy = overall_drift = 0.0
    else:
        overall_visibility = sum(vis_list) / len(vis_list)
        overall_accuracy = sum(acc_list) / len(acc_list)
        overall_drift = sum(drift_list) / len(drift_list)

    # Build a compact breakdown string for Notion
    breakdown_parts = []
    for name, s in per_llm.items():
        breakdown_parts.append(
            f"{name}: v={s['visibility']:.2f}, a={s['accuracy']:.2f}, d={s['drift']:.2f}"
        )
    breakdown_str = " | ".join(breakdown_parts)

    scores = {
        "visibility": round(overall_visibility, 3),
        "accuracy": round(overall_accuracy, 3),
        "drift": round(overall_drift, 3),
        "breakdown": breakdown_str,
        "event_label": "Weekly optimization cycle (multi-LLM)",
        "notes_joined": " | ".join(notes_chunks),
    }
    return scores


# --------------------------------------------------------------------
# Main cycle
# --------------------------------------------------------------------

def run_cycle():
    clients = load_clients()

    for c in clients:
        name = c.get("name", "Unknown")
        print(f"\n[LMO] Running multi-LLM optimization cycle for: {name}")

        prompt = build_prompt(c)
        scores = score_with_all_llms(prompt)

        page_id = c.get("notion_page_id")
        if not page_id:
            print(f"[LMO] Skipping {name}: missing notion_page_id")
            continue

        notion_update(page_id, scores)

    print("\n[LMO] Multi-LLM optimization cycle complete.")


if __name__ == "__main__":
    run_cycle()



if __name__ == "__main__":
    run_cycle()

