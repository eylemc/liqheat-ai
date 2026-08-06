from pathlib import Path
import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(".env"))

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_SECRET_KEY"]

sb = create_client(url, key)

print("CONNECTED")
print()

tables = [
    "liq_levels",
    "liq_topology",
    "liq_topology_v2",
    "liq_snapshots",
]

for table in tables:

    try:

        result = (
            sb.table(table)
            .select("*")
            .limit(1)
            .execute()
        )

        print("=" * 80)
        print(table)
        print("=" * 80)

        if result.data:
            print("ROWS:", len(result.data))
            print("COLUMNS:")
            print(list(result.data[0].keys()))
        else:
            print("EMPTY")

        print()

    except Exception as e:

        print("=" * 80)
        print(table)
        print("=" * 80)
        print("ERROR:", e)
        print()

