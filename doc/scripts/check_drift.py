import os
import sys
import requests

NOTION_TOKEN = os.environ["NOTION_TOKEN"].strip()
NOTION_DB = os.environ["NOTION_LMO_DB"].strip()

NOTION_URL = f"https://api.notion.com/v1/databases/{NOTION_DB}/query"


def query_high_drift(threshold: float = 0.4):
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

    payload = {
        "filter": {
            "and": [
                {
                    "property": "Status",
                    "select": {"does_not_equal": "System"},
                },
                {
                    "property": "Drift Score",
                    "number": {"greater_than_or_equal_to": threshold},
                },
            ]
        }
    }

    resp = requests.post(NOTION_URL, headers=headers, json=payload)
    print(f"[LMO] Notion query status: {resp.status_code}")
    if not resp.ok:
        print(resp.text)
        resp.raise_for_status()

    data = resp.json()
    return data.get("results", [])


def main():
    threshold = 0.4
    results = query_high_drift(threshold)

    if not results:
        print(f"[LMO] No clients above drift threshold ({threshold}).")
        sys.exit(0)

    print(f"[LMO] Clients above drift threshold ({threshold}):")
    for page in results:
        props = page["properties"]
        name_parts = props["Name"]["title"]
        name = name_parts[0]["plain_text"] if name_parts else "Untitled"

        drift = props.get("Drift Score", {}).get("number")
        monthly = props.get("Monthly Fee", {}).get("number")

        print(f"- {name}: drift={drift}, monthly={monthly}")

    # Exit with non-zero so GitHub marks the run as failed (alert)
    sys.exit(1)


if __name__ == "__main__":
    main()
