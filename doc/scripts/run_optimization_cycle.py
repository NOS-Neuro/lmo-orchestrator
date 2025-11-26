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

OPENAI_CLIENT = OpenAI()  # uses OPENAI_API_KEY from env
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# --------------------------------------------------------------------
# LLM configuration
# --------------------------------------------------------------------
"""
Each entry here represents a separate "LLM" for scoring.

Backends:
- openai   -> OpenAI Chat Completions API
- anthropic -> Anthropic Messages API (Claude)
- gemini    -> Google Gemini Generative Language API
"""

LLM_CONFIGS = [
    {"name": "chatgpt-4.1-mini", "backend": "openai", "model": "gpt-4.1-mini"},
    {"name": "chatgpt-4.1", "backend": "openai", "model": "gpt-4.1"},
    {"name": "claude-3.5-sonnet", "backend": "anthropic", "model": "claude-3.5-sonnet"},
    {"name": "gemini-1.5-flash", "backend": "gemini", "model": "models/gemini-1.5-flash"},
]


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def load_clients():
    path = "doc/clients/clients.yaml"
    print(f"[LMO] Loading clients from: {os.path.abspath(path)}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "clients" not in data or not isinstance(data["clients"], list):
        raise ValueError(f"Unexpected clients.yaml structure: {data}")

    print(f"[LMO] Loaded {len(data['clients'])} client(s)")
    return data["clients"]


def notion_update(page_id: str, scores: dict):
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
  "visibility": 0.0-1.0,
  "accuracy": 0.0-1.0,
  "drift": 0.0-1.0,
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
    backend = config["backend"]
    model = config["model"]
    name = config["name"]

    print(f"[LMO] Scoring with {name} ({backend}:{model})")

    # ---------- OpenAI ----------
    if backend == "openai":
        try:
            response = OPENAI_CLIENT.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},  # forces JSON
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an LMO scoring engine. "
                            "You MUST respond with a single JSON object only, "
                            "no prose, no explanation outside JSON."
                        ),
                    },
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

            return {"visibility": visibility, "accuracy": accuracy, "drift": drift, "notes": notes}

        except Exception as e:
            print(f"[LMO] Error calling {name}: {e}")
            return {
                "visibility": 0.0,
                "accuracy": 0.0,
                "drift": 0.0,
                "notes": f"Fallback scores for {name} due to error: {e}",
            }

    # (leave the anthropic + gemini branches below as-is)
    ...

    # ---------- Anthropic (Claude) ----------
    if backend == "anthropic":
        if not ANTHROPIC_API_KEY:
            print(f"[LMO] Skipping {name}: ANTHROPIC_API_KEY not set")
            return {
                "visibility": 0.0,
                "accuracy": 0.0,
                "drift": 0.0,
                "notes": f"Skipped {name}: missing ANTHROPIC_API_KEY",
            }

        try:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": model,
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are an LMO scoring engine that returns strict JSON."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    },
                ],
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data["content"][0]["text"].strip()
            print(f"[LMO] Raw response from {name}: {content}")
            parsed = json.loads(content)

            visibility = clamp01(float(parsed.get("visibility", 0.0)))
            accuracy = clamp01(float(parsed.get("accuracy", 0.0)))
            drift = clamp01(float(parsed.get("drift", 0.0)))
            notes = str(parsed.get("notes", "")).strip()

            return {"visibility": visibility, "accuracy": accuracy, "drift": drift, "notes": notes}

        except Exception as e:
            print(f"[LMO] Error calling {name}: {e}")
            return {
                "visibility": 0.0,
                "accuracy": 0.0,
                "drift": 0.0,
                "notes": f"Fallback scores for {name} due to error: {e}",
            }

    # ---------- Gemini ----------
    if backend == "gemini":
        if not GEMINI_API_KEY:
            print(f"[LMO] Skipping {name}: GEMINI_API_KEY not set")
            return {
                "visibility": 0.0,
                "accuracy": 0.0,
                "drift": 0.0,
                "notes": f"Skipped {name}: missing GEMINI_API_KEY",
            }

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={GEMINI_API_KEY}"
            full_prompt = "You are an LMO scoring engine that returns strict JSON.\n\n" + prompt
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": full_prompt}
                        ]
                    }
                ]
            }
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            print(f"[LMO] Raw response from {name}: {content}")
            parsed = json.loads(content)

            visibility = clamp01(float(parsed.get("visibility", 0.0)))
            accuracy = clamp01(float(parsed.get("accuracy", 0.0)))
            drift = clamp01(float(parsed.get("drift", 0.0)))
            notes = str(parsed.get("notes", "")).strip()

            return {"visibility": visibility, "accuracy": accuracy, "drift": drift, "notes": notes}

        except Exception as e:
            print(f"[LMO] Error calling {name}: {e}")
            return {
                "visibility": 0.0,
                "accuracy": 0.0,
                "drift": 0.0,
                "notes": f"Fallback scores for {name} due to error: {e}",
            }

    # ---------- Unsupported backend ----------
    print(f"[LMO] Unsupported backend: {backend}")
    return {
        "visibility": 0.0,
        "accuracy": 0.0,
        "drift": 0.0,
        "notes": f"Unsupported backend {backend}",
    }


def score_with_all_llms(prompt: str) -> dict:
    per_llm = {}
    vis_list, acc_list, drift_list = [], [], []

    for cfg in LLM_CONFIGS:
        name = cfg["name"]
        scores = call_llm(cfg, prompt)
        per_llm[name] = scores

        vis_list.append(scores["visibility"])
        acc_list.append(scores["accuracy"])
        drift_list.append(scores["drift"])

    if not vis_list:
        overall_visibility = overall_accuracy = overall_drift = 0.0
    else:
        overall_visibility = sum(vis_list) / len(vis_list)
        overall_accuracy = sum(acc_list) / len(acc_list)
        overall_drift = sum(drift_list) / len(drift_list)

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


