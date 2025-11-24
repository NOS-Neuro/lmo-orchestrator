import os
import json
import requests
import yaml
from datetime import datetime
from openai import OpenAI

# Environment variables
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DB = os.environ["NOTION_LMO_DB"].strip()
OPENAI_KEY = os.environ["OPENAI_API_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_KEY)

def load_clients():
    path = "doc/clients/clients.yaml"
    print(f"Loading clients from: {os.getcwd()}/{path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"clients.yaml not found at: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict) and "clients" in data:
        return data["clients"]

    if isinstance(data, list):
        return data

    raise ValueError(f"Unexpected clients.yaml structure: {data}")

def notion_update(page_id, scores):
    """Update Notion properties."""
    url = f"https://api.notion.com/v1/pages/{page_id}"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

    body = {
        "properties": {
            "Visibility Score": {"number": scores["visibility"]},
            "Accuracy Score": {"number": scores["accuracy"]},
            "Drift Score": {"number": scores["drift"]},
            "Automation Event": {
                "rich_text": [{"text": {"content": "Weekly optimization cycle"}}]
            },
            "Automation Timestamp": {
                "date": {"start": datetime.utcnow().isoformat()}
            }
        }
    }

    requests.patch(url, headers=headers, json=body)

from openai import OpenAI, RateLimitError

# ... existing setup ...

def run_cycle():
    clients = load_clients()
    print(f"[LMO] Loaded {len(clients)} client(s).")

    for c in clients:
        print(f"[LMO] Processing client: {c.get('name')}")

        prompt = f"""
        Evaluate the LLM visibility, accuracy, and drift for this company.

        Company name: {c.get('name')}
        Industry: {c.get('industry')}

        Canonical facts:
        {c.get('canonical_facts')}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an analyst that evaluates how well language models represent this company.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            # TODO: parse the response into scores
            scores = {
                "visibility": 0.75,
                "accuracy": 0.80,
                "drift": 0.10,
            }

        except RateLimitError as e:
            print(f"[LMO] Rate limit / quota issue: {e}. Using fallback scores.")
            scores = {
                "visibility": 0.60,
                "accuracy": 0.70,
                "drift": 0.20,
            }

        page_id = c.get("notion_page_id")
        if not page_id:
            raise ValueError(f"[LMO] Client {c.get('name')} is missing 'notion_page_id'")

        notion_update(page_id, scores)


if __name__ == "__main__":
    run_cycle()

