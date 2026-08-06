from dotenv import load_dotenv
from supabase import create_client
import os

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SECRET_KEY")

print("URL found:", bool(url))
print("KEY found:", bool(key))

if not url or not key:
    raise RuntimeError("SUPABASE_URL veya SUPABASE_SECRET_KEY okunamadı")

sb = create_client(url, key)
buckets = sb.storage.list_buckets()

print("\nBuckets:")
for b in buckets:
    if isinstance(b, dict):
        print("-", b.get("name"))
    else:
        print("-", getattr(b, "name", str(b)))
