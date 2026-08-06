import json
import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

sb = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SECRET_KEY"],
)

result = (
    sb.table("liq_logging")
    .select("*")
    .order("logged_at", desc=False)
    .limit(3)
    .execute()
)

print("Örnek satır sayısı:", len(result.data))

for index, row in enumerate(result.data, start=1):
    print(f"\n=== ROW {index} ===")
    print("Kolonlar:", list(row.keys()))

    for key, value in row.items():
        if key == "payload":
            print("\npayload tipi:", type(value).__name__)

            if isinstance(value, dict):
                print("payload keys:", list(value.keys()))

            preview = json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )

            print("payload preview:")
            print(preview[:3000])
        else:
            text = str(value)
            print(f"{key}: {text[:500]}")
