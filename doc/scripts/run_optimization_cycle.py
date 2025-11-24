import os
import json
import requests
import yaml
from datetime import datetime
from openai import OpenAI

# Environment variables
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DB = os.environ["NOTION_LMO_DB"]
OPENAI_KEY = os.environ["OPENAI_API_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_KEY)

def load_clients():
    """Load client definitions from YAML."""
    with open("doc/clients/clients.yaml", "r") as f:
        data = yaml.safe_load(f)
    return data["clients"]

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

def run_cycle():
    """Main optimization routine."""
    clients = load_clients()

    for c in clients:
        prompt = f"""
        Evaluate the LLM visibility, accuracy, and drift for:
        Company: {c['name']}
        Canonical facts:
        {c['canonical_facts']}
        """

        # New OpenAI SDK format
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Evaluate company representation."},
                {"role": "user", "content": prompt}
            ]
        )

        # For now: placeholder deterministic scores
        scores = {
            "visibility": 0.75,
            "accuracy": 0.80,
            "drift": 0.10
        }

        notion_update(c["notion_page_id"], scores)

if __name__ == "__main__":
    run_cycle()

