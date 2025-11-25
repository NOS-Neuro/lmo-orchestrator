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
# Helpers
# --------------------------------------------------------------------

def load_clients():
    """
    Load clients from doc/clients/clients.yaml

    Expected structure:

    clients:
      - name: "Acme Logistics"
        slug: "acme-logistics"
        location: "Toronto, ON, Canada"
        industry: "3PL / warehousing"
        canonical_facts: |
          - Fact 1
          - Fact 2
        notion_page_id: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
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

    This assumes the database has the following properties:
      - Visibility Score (number)
      - Accuracy Score   (number)
      - Drift Score      (number)
      - Automation Event (rich text)
      - Automation Timestamp (date)
      - Last Optimization Date (date)
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
                    {"text": {"content": scores.get("event_label", "Weekly optimization cycle")}}
                ]
            },
            "Automation Timestamp": {"date": {"start": now_iso}},
            "Last Optimization Date": {"date": {"start": now_iso}},
            # If you later add a "Latest Notes" rich text property, you can uncomment:
            # "Latest Notes": {
            #     "rich_text": [{"text": {"content": scores.get("notes", "")[:1900]}}],
            # },
        }
    }

    print(f"[LMO] Updating Notion page {page_id} with scores: {scores}")
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


def call_openai_for_scores(prompt: str) -> dict:
    """
    Call OpenAI to get visibility/accuracy/drift scores as JSON.
    Includes a defensive fallback if parsing fails.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an LMO scoring engine that returns strict JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()
        print(f"[LMO] Raw model response: {content}")

        data = json.loads(content)

        visibility = float(data.get("visibility", 0.0))
        accuracy = float(data.get("accuracy", 0.0))
        drift = float(data.get("drift", 0.0))
        notes = str(data.get("notes", "")).strip()

        # Clamp values to 0–1 to avoid weird outputs
        def clamp(x):
            return max(0.0, min(1.0, x))

        scores = {
            "visibility": clamp(visibility),
            "accuracy": clamp(accuracy),
            "drift": clamp(drift),
            "notes": notes,
            "event_label": "Weekly optimization cycle",
        }
        return scores

    except Exception as e:
        print(f"[LMO] Error calling OpenAI or parsing JSON: {e}")
        # Safe fallback: neutral scores with error note
        return {
            "visibility": 0.0,
            "accuracy": 0.0,
            "drift": 0.0,
            "notes": f"Fallback scores due to error: {e}",
            "event_label": "Weekly optimization cycle (fallback)",
        }


# --------------------------------------------------------------------
# Main cycle
# --------------------------------------------------------------------

def run_cycle():
    clients = load_clients()

    for c in clients:
        name = c.get("name", "Unknown")
        print(f"\n[LMO] Running optimization cycle for: {name}")

        prompt = build_prompt(c)
        scores = call_openai_for_scores(prompt)

        page_id = c.get("notion_page_id")
        if not page_id:
            print(f"[LMO] Skipping {name}: missing notion_page_id")
            continue

        notion_update(page_id, scores)

    print("\n[LMO] Optimization cycle complete.")


if __name__ == "__main__":
    run_cycle()

