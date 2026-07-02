"""File I/O utilities — eliminates repeated os.makedirs + write patterns."""

import csv
import json
import os


def ensure_parent_dir(path):
    """Create parent directories for *path* if they don't exist."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    return parent


def write_json(data, path):
    """Serialize *data* as pretty-printed JSON to *path*, creating dirs as needed."""
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv_report(report, csv_path):
    """Flatten key metrics per page into a CSV file."""
    ensure_parent_dir(csv_path)
    headers = [
        "url", "http_status", "title", "title_length",
        "meta_description", "meta_description_length",
        "h1_count", "word_count", "images_total", "images_missing_alt",
        "links_count", "canonical", "hreflang_count", "json_ld_blocks",
        "score", "score_reasons",
        "pagespeed_performance",
        "moz_domain_authority", "moz_page_authority", "moz_external_links",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        pages = report.get("pages", {})
        for url, p in pages.items():
            fetch = p.get("fetch", {})
            analysis = p.get("analysis", {}) or {}
            scores = p.get("scores", {}) or {}
            pagespeed = (report.get("pagespeed") or {}).get(url) or {}
            moz = p.get("offpage", {}) or {}
            row = {
                "url": url,
                "http_status": fetch.get("status"),
                "title": _nested(analysis, "title", "text", ""),
                "title_length": _nested(analysis, "title", "length", 0),
                "meta_description": _nested(analysis, "meta_description", "text", ""),
                "meta_description_length": _nested(analysis, "meta_description", "length", 0),
                "h1_count": _nested(analysis, "h1", "count", 0),
                "word_count": analysis.get("word_count", 0),
                "images_total": _nested(analysis, "images", "total", 0),
                "images_missing_alt": _nested(analysis, "images", "missing_alt_count", 0),
                "links_count": _nested(analysis, "links", "count", 0),
                "canonical": analysis.get("canonical", ""),
                "hreflang_count": len(analysis.get("hreflangs", [])),
                "json_ld_blocks": len(analysis.get("json_ld", [])),
                "score": scores.get("score"),
                "score_reasons": "; ".join(scores.get("reasons", [])),
                "pagespeed_performance": (
                    pagespeed.get("lighthouseResult", {})
                    .get("categories", {})
                    .get("performance", {})
                    .get("score")
                    if isinstance(pagespeed, dict)
                    else None
                ),
                "moz_domain_authority": moz.get("domain_authority"),
                "moz_page_authority": moz.get("page_authority"),
                "moz_external_links": moz.get("external_links"),
            }
            writer.writerow(row)


def _nested(data, key1, key2, default=None):
    """Safely access data[key1][key2] with a fallback default."""
    return data.get(key1, {}).get(key2, default)
