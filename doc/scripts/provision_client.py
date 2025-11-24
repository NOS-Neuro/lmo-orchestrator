import os
import requests
import yaml
from datetime import datetime

# --- ENVIRONMENT ---

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DB = os.environ["NOTION_LMO_DB"].strip()
ADMIN_TOKEN = os.environ["LMO_ADMIN_TOKEN"]

GITHUB_OWNER = os.environ["GITHUB_OWNER"]
CLIENT_NAME = os.environ["CLIENT_NAME"]
CLIENT_SLUG = os.environ["CLIENT_SLUG"]
INDUSTRY = os.environ["INDUSTRY"]
MONTHLY_FEE = float(os.environ.get("MONTHLY_FEE", "0") or 0)
TEMPLATE_REPO = os.environ["TEMPLATE_REPO"]

CLIENTS_YAML_PATH = "doc/clients/clients.yaml"


def create_repo_from_template():
    """Use GitHub API to create a new repo from a template."""
    new_repo_name = f"{CLIENT_SLUG}-LMO"

    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{TEMPLATE_REPO}/generate"
    headers = {
        "Authorization": f"token {ADMIN_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    payload = {
        "owner": GITHUB_OWNER,
        "name": new_repo_name,
        "private": False,
    }

    print(f"[LMO] Creating new repo from template: {GITHUB_OWNER}/{TEMPLATE_REPO} -> {new_repo_name}")
    resp = requests.post(url, headers=headers, json=payload)
    print(f"[LMO] GitHub create repo status: {resp.status_code}")
    if not resp.ok:
        print(resp.text)
        resp.raise_for_status()

    data = resp.json()
    full_name = data["full_name"]          # e.g. "NOS-Neuro/acme-logistics-LMO"
    html_url = data["html_url"]
    print(f"[LMO] New repo created: {full_name} ({html_url})")
    return full_name, html_url


def create_notion_row():
    """Create a new row in the LMO Client Tracker database."""
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

    now_iso = datetime.utcnow().isoformat()

    body = {
        "parent": {"database_id": NOTION_DB},
        "properties": {
            "Client Name": {
                "title": [{"text": {"content": CLIENT_NAME}}],
            },
            "Industry": {
                "rich_text": [{"text": {"content": INDUSTRY}}],
            },
            "Status": {
                "select": {"name": "Active"},
            },
            "Monthly Fee": {
                "number": MONTHLY_FEE,
            },
            "Automation Event": {
                "rich_text": [{"text": {"content": "Provisioned via orchestrator"}}],
            },
            "Automation Timestamp": {
                "date": {"start": now_iso},
            },
        },
    }

    print(f"[LMO] Creating Notion row for {CLIENT_NAME}")
    resp = requests.post(url, headers=headers, json=body)
    print(f"[LMO] Notion create status: {resp.status_code}")
    if not resp.ok:
        print(resp.text)
        resp.raise_for_status()

    data = resp.json()
    page_id = data["id"]
    print(f"[LMO] Notion page id: {page_id}")
    return page_id


def update_clients_yaml(notion_page_id: str):
    """Append new client entry to doc/clients/clients.yaml."""
    print(f"[LMO] Updating {CLIENTS_YAML_PATH}")

    if os.path.exists(CLIENTS_YAML_PATH):
        with open(CLIENTS_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    if isinstance(data, list):
        # legacy structure: treat as direct list
        clients = data
        data = {"clients": clients}

    clients = data.get("clients", [])

    new_entry = {
        "name": CLIENT_NAME,
        "slug": CLIENT_SLUG,
        "industry": INDUSTRY,
        "canonical_facts": f"- {CLIENT_NAME} is a client of NOS LMO.\n- Industry: {INDUSTRY}",
        "competitors": "",
        "notion_page_id": notion_page_id,
    }

    clients.append(new_entry)
    data["clients"] = clients

    with open(CLIENTS_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"[LMO] Added client to {CLIENTS_YAML_PATH}: {CLIENT_NAME}")
    return True


def main():
    repo_full_name, repo_url = create_repo_from_template()
    notion_page_id = create_notion_row()
    did_update = update_clients_yaml(notion_page_id)

    # Expose outputs back to GitHub Actions
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"repo_full_name={repo_full_name}\n")
            f.write(f"repo_url={repo_url}\n")
            f.write(f"notion_page_id={notion_page_id}\n")
            f.write(f"did_update_clients_yaml={'true' if did_update else 'false'}\n")

    print("[LMO] Provisioning complete.")


if __name__ == "__main__":
    main()
