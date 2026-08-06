from __future__ import annotations

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SECRET_KEY")

if not url or not key:
    raise RuntimeError("Supabase ayarları bulunamadı")

sb = create_client(url, key)

TABLES = {
    "liq_logging": "logged_at",
    "liq_insight_snapshots": "snapshot_at",
    "liq_insights": "refreshed_at",
    "liquidation_cache": "updated_at",
    "ai_analysis_cache": "created_at",
    "liq_logging_archive_runs": "started_at",
}

for table, date_col in TABLES.items():
    print(f"\n=== {table} ===")

    try:
        result = (
            sb.table(table)
            .select("*", count="exact")
            .limit(1)
            .execute()
        )

        print("Satır sayısı:", result.count)

        oldest = (
            sb.table(table)
            .select(date_col)
            .order(date_col, desc=False)
            .limit(1)
            .execute()
        )

        newest = (
            sb.table(table)
            .select(date_col)
            .order(date_col, desc=True)
            .limit(1)
            .execute()
        )

        oldest_value = oldest.data[0].get(date_col) if oldest.data else None
        newest_value = newest.data[0].get(date_col) if newest.data else None

        print("İlk kayıt:", oldest_value)
        print("Son kayıt:", newest_value)

    except Exception as exc:
        print("HATA:", exc)
