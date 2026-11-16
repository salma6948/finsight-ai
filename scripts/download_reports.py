"""Download optional official 2026 annual-report PDFs; network required.

Usage: python scripts/download_reports.py --out .local/reports
Only trusted hard-coded primary source URLs are used, with byte limits and content checks.
Downloaded documents are deliberately not committed to Git.
"""
import argparse
from pathlib import Path
import requests

REPORTS = {
    "apple_2026_10k.pdf": "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/16fd0081-220e-4209-9531-d395a65dd3c7.pdf",
    "nvidia_2026_annual.pdf": "https://s201.q4cdn.com/141608511/files/doc_financials/2026/ar/2026-Annual-Report-Web.pdf",
}
MAX_BYTES = 20 * 1024 * 1024


def download(url: str, output: Path) -> None:
    with requests.get(url, timeout=(10, 45), stream=True,
                      headers={"User-Agent": "FinSightAI/1.0 (educational research; manual download)"}) as r:
        r.raise_for_status()
        chunks = []
        total = 0
        for chunk in r.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > MAX_BYTES:
                raise ValueError("PDF exceeds 20 MB limit")
            chunks.append(chunk)
        payload = b"".join(chunks)
        if not payload.startswith(b"%PDF-"):
            raise ValueError("Server did not return a PDF")
        output.write_bytes(payload)
        print(f"Downloaded {output} ({total:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(".local/reports"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, source in REPORTS.items():
        try:
            download(source, args.out / filename)
        except (requests.RequestException, ValueError) as exc:
            print(f"Could not download {filename}: {exc}")
