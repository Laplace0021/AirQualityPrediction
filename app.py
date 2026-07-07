"""
test_openaq.py
Script diagnostik berdiri sendiri untuk mencari tahu kenapa
/sensors/{id}/measurements dan /sensors/{id}/hours mengembalikan kosong,
padahal sensor & lokasinya valid.

Jalankan: python3 test_openaq.py
Lalu copy-paste SELURUH output ke chat.
"""

import os
import requests
from datetime import datetime, timedelta

OPENAQ_API_KEY = os.environ.get(
    "OPENAQ_API_KEY",
    "430a6cbeb038741241c9129a3323543b8a15f7e2b80bd32d9d07b2efb3d66aff"
)
BASE = "https://api.openaq.org/v3"
HEADERS = {"X-API-Key": OPENAQ_API_KEY}

SENSOR_ID = 14739443  # pm25, sudah dikonfirmasi valid dari /locations/6144741/sensors
LOCATION_ID = 6144741


def show(title, resp):
    print("=" * 70)
    print(title)
    print("URL     :", resp.url)
    print("STATUS  :", resp.status_code)
    print("BODY    :", resp.text[:2000])
    print()


def main():
    now = datetime.utcnow()
    dt_to = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    dt_from = (now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"API KEY (10 char pertama): {OPENAQ_API_KEY[:10]}...")
    print(f"now (UTC)  : {dt_to}")
    print(f"from (UTC) : {dt_from}")
    print()

    # 1. Sanity check: sensor detail (endpoint paling sederhana, tanpa filter tanggal)
    r1 = requests.get(f"{BASE}/sensors/{SENSOR_ID}", headers=HEADERS, timeout=15)
    show("1) GET /sensors/{id}  (detail sensor, tanpa filter apapun)", r1)

    # 2. /measurements TANPA filter tanggal sama sekali
    r2 = requests.get(
        f"{BASE}/sensors/{SENSOR_ID}/measurements",
        params={"limit": 5},
        headers=HEADERS,
        timeout=15
    )
    show("2) GET /sensors/{id}/measurements  (tanpa filter tanggal, limit=5)", r2)

    # 3. /measurements DENGAN filter tanggal 24 jam terakhir
    r3 = requests.get(
        f"{BASE}/sensors/{SENSOR_ID}/measurements",
        params={"limit": 5, "datetime_from": dt_from, "datetime_to": dt_to},
        headers=HEADERS,
        timeout=15
    )
    show("3) GET /sensors/{id}/measurements  (dengan datetime_from/to 24 jam terakhir)", r3)

    # 4. /hours TANPA filter tanggal
    r4 = requests.get(
        f"{BASE}/sensors/{SENSOR_ID}/hours",
        params={"limit": 5},
        headers=HEADERS,
        timeout=15
    )
    show("4) GET /sensors/{id}/hours  (tanpa filter tanggal, limit=5)", r4)

    # 5. /hours DENGAN filter tanggal 24 jam terakhir
    r5 = requests.get(
        f"{BASE}/sensors/{SENSOR_ID}/hours",
        params={"limit": 5, "datetime_from": dt_from, "datetime_to": dt_to},
        headers=HEADERS,
        timeout=15
    )
    show("5) GET /sensors/{id}/hours  (dengan datetime_from/to 24 jam terakhir)", r5)

    # 6. /locations/{id}/latest -> endpoint ringkas utk nilai terbaru semua sensor lokasi
    r6 = requests.get(
        f"{BASE}/locations/{LOCATION_ID}/latest",
        headers=HEADERS,
        timeout=15
    )
    show("6) GET /locations/{id}/latest", r6)


if __name__ == "__main__":
    main()
