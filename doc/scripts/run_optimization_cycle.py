import os
import datetime as dt
import json
import requests
import yaml

# === Config from environment ===
NOTION_API = "https://api.notion.com/v1/pages"
NOTION_VERSION = "2022-06-28"
OPENAI_API = "https://api.openai.com/v1/chat/completions"

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def load_clients(config_path: str = "clients/clients.yaml"):
    """Load client definitions from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("clients", [])


def call_openai(prompt: str, model: str = "gpt-4.1-mini") -> str:
    """Call OpenAI and return the assistant message content."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an evaluator of how well AI systems describe companies. "
                    "You score visibility, accuracy, and drift based on given facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }
    resp = requests.post(OPENAI_API, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return content


def build_evaluation_prompt(client: dict) -> str:
    """
    Ask the model to simulate how LLMs represent this company and to self-score:
      - visibility_score
      - accuracy_score
      - drift_score
    """
    primary_q = client["primary_query"]
    company_q = client["company_query"]
    canonical = client["canonical_facts"]
    competitors = client["competitors"]

    return f"""
You are evaluating how well large language models represent a company in real-world queries.

Company: {client["name"]}
Location: {client["location"]}
Industry: {client["industry"]}

Canonical facts about this company:
{canonical}

Key competitors in this space:
{competitors}

TASK:

1. Imagine a typical user asks an AI assistant the question:
   "{primary_q}"

2. Then imagine the user asks:
   "{company_q}"

3. Based on your knowledge and the canonical facts provided, rate the following from 0 to 100:

   - visibility_score: How likely is this company to appear prominently and positively in answers to the primary query (0 = never, 100 = always one of the top recommended)?
   - accuracy_score: How accurate are typical AI descriptions of this company vs the canonical facts (0 = mostly wrong, 100 = fully accurate)?
   - drift_score: How much drift or misalignment is there between typical AI answers and the canonical facts (0 = extreme drift, 100 = perfectly aligned)?

4. Return ONLY a single JSON object with this exact structure:

{{
  "visibility_score": <number>,
  "accuracy_score": <number>,
  "drift_score": <number>,
  "notes": "<short explanation of why you chose these scores>"
}}
"""


def parse_scores(raw: str) -> dict:
    """Extract JSON from the model output safely."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "visibility_score": 0,
            "accuracy_score": 0,
            "drift_score": 0,
            "notes": f"Failed to parse JSON from: {raw[:200]}...",
        }
    return data


def update_notion_client_row(client: dict, scores: dict):
    """Patch the specific client page in Notion with new scores and dates."""
    page_id = client["notion_page_id"]
    now_iso = dt.datetime.utcnow().isoformat() + "Z"

    properties = {
        "Visibility Score": {"number": float(scores.get("visibility_score", 0))},
        "Accuracy Score": {"number": float(scores.get("accuracy_score", 0))},
        "Drift Score": {"number": float(scores.get("drift_score", 0))},
        "Last Optimization Date": {"date": {"start": now_iso}},
        "Automation Event": {
            "rich_text": [
                {
                    "text": {
                        "content": (
                            "Monthly optimization run – "
                            f"visibility={scores.get('visibility_score')}, "
                            f"accuracy={scores.get('accuracy_score')}, "
                            f"drift={scores.get('drift_score')}"
                        )
                    }
                }
            ]
        },
        "Automation Timestamp": {"date": {"start": now_iso}},
    }

    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }
    body = {"properties": properties}

    resp = requests.patch(f"{NOTION_API}/{page_id}", headers=headers, json=body, timeout=60)
    resp.raise_for_status()


def run_cycle():
    clients = load_clients()
    print(f"Loaded {len(clients)} clients from clients/clients.yaml")

    for client in clients:
        print(f"Running optimization for {client['name']} ({client['slug']})")
        prompt = build_evaluation_prompt(client)
        raw = call_openai(prompt)
        scores = parse_scores(raw)
        print(f"Scores for {client['name']}: {scores}")
        update_notion_client_row(client, scores)
        print(f"Updated Notion row for {client['name']}")


if __name__ == "__main__":
    run_cycle()
