"""
SignBridge AI — Asset Downloader (download_assets.py)

Downloads static assets required for offline operation:
  - Three.js r128 (3D avatar rendering)

Run once during setup:
    python scripts/download_assets.py
"""

import os
import sys
import hashlib
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS_DIR = os.path.join(BASE_DIR, "frontend", "assets", "js")

ASSETS = [
    {
        "name": "three.min.js",
        "url": "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
        "dest": os.path.join(JS_DIR, "three.min.js"),
        # SHA-512 of the unmodified file from cdnjs — verify after download
        "sha512": None,  # Set after first verified download
    },
]


def download(url: str, dest: str, timeout: int = 30):
    print(f"  Downloading: {url}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    size = os.path.getsize(dest)
    print(f"  Saved: {dest} ({size:,} bytes)")
    return dest


def sha512_of_file(path: str) -> str:
    h = hashlib.sha512()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    import base64
    return "sha512-" + base64.b64encode(h.digest()).decode()


def main():
    os.makedirs(JS_DIR, exist_ok=True)
    success = True

    for asset in ASSETS:
        dest = asset["dest"]
        if os.path.exists(dest) and os.path.getsize(dest) > 10000:
            print(f"  [SKIP] {asset['name']} already exists.")
            sri = sha512_of_file(dest)
            print(f"         SRI: {sri}")
            continue

        try:
            download(asset["url"], dest)
            sri = sha512_of_file(dest)
            print(f"  SRI hash: {sri}")
            print(f"  Add this to your <script integrity=\"{sri}\" ...> tag if needed.")
        except Exception as e:
            print(f"  [ERROR] Failed to download {asset['name']}: {e}")
            print(f"  Manually download from: {asset['url']}")
            print(f"  Save to: {dest}")
            success = False

    if success:
        print("\n[OK] All assets downloaded. SignBridge AI is ready for offline use.")
    else:
        print("\n[WARNING] Some assets could not be downloaded. See instructions above.")


if __name__ == "__main__":
    main()
