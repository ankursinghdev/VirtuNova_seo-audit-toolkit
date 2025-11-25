#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VirtuNova SEO Audit Toolkit — Full on-page, off-page, technical audit (Hybrid Playwright + PageSpeed)
Outputs: JSON, CSV, PDF, web_ui/report.json

Usage example:
  python seo_audit_tool_extended.py \
    --url https://www.ellocentlabs.com \
    --output reports/report.json \
    --pages 100 \
    --pagespeed-key $PAGESPEED_KEY \
    --moz-access-id $MOZ_ACCESS_ID \
    --moz-secret $MOZ_SECRET \
    --web-ui

Notes:
- For off-page metrics (Domain Authority, Page Authority, backlinks) provide Moz API keys
  (or Ahrefs/SEM tools if you want to add them - code can be extended).
- Playwright must be installed and Chromium available:
    pip install -r requirements.txt
    pip install playwright
    playwright install chromium
"""
import argparse
import asyncio
import csv
import hashlib
import hmac
import json
import os
import re
import time
import math
from collections import deque
from datetime import datetime
from urllib.parse import urljoin, urlparse

# Optional libs loaded at runtime
try:
    from playwright.async_api import async_playwright, TimeoutError as PWTimeout
except Exception:
    async_playwright = None
    PWTimeout = Exception

try:
    import aiohttp
except Exception:
    aiohttp = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib import colors
    from reportlab.lib.units import inch
except Exception:
    SimpleDocTemplate = None

# ---------- Config ----------
USER_AGENT = "VirtuNova-SEO-Toolkit/1.0 (+https://virtunova.com)"
DEFAULT_MAX_PAGES = 100
REQUEST_TIMEOUT = 30
PLAYWRIGHT_TIMEOUT_MS = 30000

# ---------- Helpers ----------
def normalize_url(base, href):
    if not href: return None
    href = href.strip()
    if href.startswith(("javascript:", "mailto:", "#")): return None
    joined = urljoin(base, href)
    parsed = urlparse(joined)
    cleaned = parsed._replace(fragment="")
    return cleaned.geturl()

def same_origin(a, b):
    pa, pb = urlparse(a), urlparse(b)
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)

# ---------- Playwright renderer ----------
async def render_page_content(url, timeout=PLAYWRIGHT_TIMEOUT_MS):
    if not async_playwright:
        return {"url": url, "status": None, "error": "playwright-not-installed"}
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"], headless=True)
            page = await browser.new_page(user_agent=USER_AGENT)
            try:
                resp = await page.goto(url, timeout=timeout, wait_until="networkidle")
            except PWTimeout:
                try:
                    resp = await page.goto(url, timeout=timeout, wait_until="load")
                except Exception as e:
                    await browser.close()
                    return {"url": url, "status": None, "error": f"navigation-timeout: {e}"}
            status = resp.status if resp else None
            # allow dynamic content settle
            try:
                await page.wait_for_timeout(500)
            except Exception:
                pass
            content = await page.content()
            title = ""
            try:
                title = await page.title()
            except Exception:
                title = ""
            headers = {}
            try:
                if resp:
                    headers = dict(await resp.all_headers())
            except Exception:
                headers = {}
            await browser.close()
            return {"url": url, "status": status, "content": content, "title": title, "headers": headers}
    except Exception as e:
        return {"url": url, "status": None, "error": str(e)}

# ---------- HTML analysis ----------
def analyze_html_from_text(url, html):
    if not BeautifulSoup or not html:
        return {}
    soup = BeautifulSoup(html, "lxml")
    result = {}
    # title & meta
    title_tag = soup.find("title")
    title = title_tag.string.strip() if title_tag and title_tag.string else ""
    result["title"] = {"text": title, "length": len(title)}
    desc_tag = soup.find("meta", attrs={"name": re.compile("description", re.I)})
    desc = desc_tag["content"].strip() if desc_tag and desc_tag.get("content") else ""
    result["meta_description"] = {"text": desc, "length": len(desc)}
    # headings
    h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]
    result["h1"] = {"count": len(h1s), "texts": h1s}
    # canonical
    can_tag = soup.find("link", rel=re.compile("canonical", re.I))
    canonical = can_tag["href"].strip() if can_tag and can_tag.get("href") else ""
    result["canonical"] = canonical
    # robots, viewport
    robots_tag = soup.find("meta", attrs={"name": re.compile("robots", re.I)})
    robots = robots_tag["content"].strip() if robots_tag and robots_tag.get("content") else ""
    result["meta_robots"] = robots
    viewport_tag = soup.find("meta", attrs={"name": re.compile("viewport", re.I)})
    viewport = viewport_tag["content"].strip() if viewport_tag and viewport_tag.get("content") else ""
    result["viewport"] = viewport
    # json-ld
    json_ld = []
    for s in soup.find_all("script", type="application/ld+json"):
        if s.string:
            try:
                json_ld.append(json.loads(s.string))
            except Exception:
                json_ld.append({"raw": s.string[:500]})
    result["json_ld"] = json_ld
    # images
    imgs = soup.find_all("img")
    imgs_missing_alt = [i.get("src") for i in imgs if not i.get("alt")]
    result["images"] = {"total": len(imgs), "missing_alt_count": len(imgs_missing_alt), "missing_alt_srcs": imgs_missing_alt[:50]}
    # links & word count
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    result["links"] = {"count": len(links)}
    body = soup.body
    if body:
        body_text = body.get_text(separator=" ", strip=True)
        words = re.findall(r"\w+", body_text)
        result["word_count"] = len(words)
    else:
        result["word_count"] = 0
    # hreflang
    hreflangs = []
    for link in soup.find_all("link", attrs={"rel": re.compile("alternate", re.I)}):
        hreflang = link.get("hreflang")
        href = link.get("href")
        if hreflang and href:
            hreflangs.append({"hreflang": hreflang, "href": href})
    result["hreflangs"] = hreflangs
    # detect canonical chain? left to separate function
    return result

def validate_json_ld(json_ld_blocks):
    issues = []
    for i, b in enumerate(json_ld_blocks):
        if isinstance(b, dict):
            if "@context" not in b and "@graph" not in b:
                issues.append({"index": i, "issue": "missing @context"})
            if "@type" not in b and "@graph" not in b:
                issues.append({"index": i, "issue": "missing @type"})
        else:
            issues.append({"index": i, "issue": "not a dict"})
    return issues

def canonical_chain_check(pages):
    chains = []
    for url, data in pages.items():
        c = data.get("analysis", {}).get("canonical")
        if not c:
            continue
        chain = [url]
        nxt = c
        while nxt and nxt != chain[-1] and nxt in pages and len(chain) < 20:
            chain.append(nxt)
            nxt = pages[nxt].get("analysis", {}).get("canonical")
        if len(chain) > 1:
            chains.append(chain)
    return chains

# ---------- PageSpeed Insights ----------
async def pagespeed_insights(url, api_key, strategy="mobile"):
    if not api_key:
        return {"error": "no_api_key"}
    if not aiohttp:
        return {"error": "aiohttp-not-installed"}
    api = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    params = {"url": url, "key": api_key, "strategy": strategy}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(api, params=params, timeout=30) as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}

# ---------- Moz (Off-page metrics) ----------
def create_moz_auth(access_id, secret):
    # returns (url, headers) ready for GET to Moz API v2 (linkscape has been deprecated)
    # We will use Moz's "url-metrics" style signature if you have access.
    # NOTE: if you do not have Moz keys, this function will not be used.
    expires = int(time.time()) + 300
    string_to_sign = f"{access_id}\n{expires}"
    h = hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1)
    signature = h.digest()
    signature_b64 = signature.hex()  # different encodings used by older examples; confirm with Moz docs
    return expires, signature_b64

def fetch_moz_metrics(url, access_id, secret):
    # Placeholder implementation. Moz's modern API uses OAuth or their new endpoints.
    # If you have MOZ access_id & secret, you can implement exact call per Moz docs.
    # Here we return Nones if keys not provided.
    if not access_id or not secret:
        return {"domain_authority": None, "page_authority": None, "external_links": None, "moz_rank": None}
    # In practice implement per docs: https://moz.com/help/links-api
    # We'll return stub telling user to fill in their API access implementation.
    return {"domain_authority": None, "page_authority": None, "external_links": None, "note": "provide MOZ keys and implement fetch_moz_metrics"}

# ---------- Crawler (Playwright-based) ----------
class PlaywrightCrawler:
    def __init__(self, seed_url, max_pages=DEFAULT_MAX_PAGES):
        self.seed = seed_url
        self.seed_netloc = urlparse(seed_url).netloc
        self.to_visit = deque([seed_url])
        self.seen = set([seed_url])
        self.results = {}
        self.max_pages = max_pages

    async def run(self):
        if not async_playwright:
            raise RuntimeError("Playwright is required. Install 'playwright' and run 'playwright install chromium'.")
        sem = asyncio.Semaphore(3)
        async def worker():
            while self.to_visit and len(self.results) < self.max_pages:
                url = self.to_visit.popleft()
                async with sem:
                    page_result = await render_page_content(url)
                    self.results[url] = {"fetch": page_result}
                    if page_result.get("status") and page_result.get("content"):
                        analysis = analyze_html_from_text(url, page_result["content"])
                        self.results[url]["analysis"] = analysis
                        # extract links and enqueue same-origin pages
                        try:
                            soup = BeautifulSoup(page_result["content"], "lxml")
                            for a in soup.find_all("a", href=True):
                                n = normalize_url(url, a["href"])
                                if not n: continue
                                if urlparse(n).netloc == self.seed_netloc and n not in self.seen and len(self.seen) < self.max_pages:
                                    self.seen.add(n); self.to_visit.append(n)
                        except Exception:
                            pass
        tasks = [asyncio.create_task(worker()) for _ in range(3)]
        await asyncio.gather(*tasks)

# ---------- Scoring ----------
def score_page(analysis, fetch_info):
    score = 100
    reasons = []
    tlen = analysis.get("title", {}).get("length", 0)
    if tlen == 0:
        score -= 20; reasons.append("Missing title")
    if analysis.get("meta_description", {}).get("length", 0) == 0:
        score -= 10; reasons.append("Missing meta description")
    if analysis.get("h1", {}).get("count", 0) == 0:
        score -= 10; reasons.append("Missing H1")
    if analysis.get("word_count", 0) < 100:
        score -= 5; reasons.append("Low word count (<100)")
    status = fetch_info.get("status")
    if status is None or (status and status >= 400):
        score = 0; reasons.append(f"HTTP error: {status}")
    return {"score": max(0, score), "reasons": reasons}

# ---------- CSV export ----------
def write_csv_report(report, csv_path):
    # Flatten key metrics per page into CSV rows
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    headers = [
        "url", "http_status", "title", "title_length", "meta_description", "meta_description_length",
        "h1_count", "word_count", "images_total", "images_missing_alt", "links_count",
        "canonical", "hreflang_count", "json_ld_blocks", "score", "score_reasons",
        "pagespeed_performance", "moz_domain_authority", "moz_page_authority", "moz_external_links"
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
                "title": analysis.get("title", {}).get("text", ""),
                "title_length": analysis.get("title", {}).get("length", 0),
                "meta_description": analysis.get("meta_description", {}).get("text", ""),
                "meta_description_length": analysis.get("meta_description", {}).get("length", 0),
                "h1_count": analysis.get("h1", {}).get("count", 0),
                "word_count": analysis.get("word_count", 0),
                "images_total": analysis.get("images", {}).get("total", 0),
                "images_missing_alt": analysis.get("images", {}).get("missing_alt_count", 0),
                "links_count": analysis.get("links", {}).get("count", 0),
                "canonical": analysis.get("canonical", ""),
                "hreflang_count": len(analysis.get("hreflangs", [])),
                "json_ld_blocks": len(analysis.get("json_ld", [])),
                "score": scores.get("score"),
                "score_reasons": "; ".join(scores.get("reasons", [])),
                "pagespeed_performance": pagespeed.get("lighthouseResult", {}).get("categories", {}).get("performance", {}).get("score") if isinstance(pagespeed, dict) else None,
                "moz_domain_authority": moz.get("domain_authority"),
                "moz_page_authority": moz.get("page_authority"),
                "moz_external_links": moz.get("external_links")
            }
            writer.writerow(row)

# ---------- PDF report ----------
def generate_pdf_report(report_data, output_path="reports/SEO_Audit_Report.pdf"):
    if SimpleDocTemplate is None:
        print("PDF generation skipped — reportlab not installed.")
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    primary = colors.HexColor("#A020F0")
    accent = colors.HexColor("#E9407A")
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], textColor=primary, fontSize=20)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], textColor=accent, fontSize=14)
    normal = styles["Normal"]
    elems = []
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        try:
            elems.append(Image(logo_path, width=1.6*inch, height=1.6*inch))
            elems.append(Spacer(1, 0.15*inch))
        except Exception:
            pass
    elems.append(Paragraph("VirtuNova — Full SEO Audit Report", title_style))
    elems.append(Spacer(1, 0.1*inch))
    elems.append(Paragraph(f"<b>Website:</b> {report_data.get('site','')}", normal))
    elems.append(Paragraph(f"<b>Generated:</b> {report_data.get('generated_at','')}", normal))
    elems.append(Spacer(1, 0.15*inch))
    pages = report_data.get("pages", {})
    total_pages = len(pages)
    elems.append(Paragraph("Executive Summary", heading_style))
    elems.append(Paragraph(f"Total pages crawled: {total_pages}", normal))
    scores = [p.get("scores", {}).get("score") for p in pages.values() if p.get("scores")]
    avg = round(sum(scores)/max(1,len(scores)),1) if scores else 0
    elems.append(Paragraph(f"Average SEO Score: <b>{avg}%</b>", normal))
    elems.append(Spacer(1, 0.15*inch))
    # Top issues
    issues = [[url, ", ".join(p.get("scores", {}).get("reasons", []))] for url,p in pages.items() if p.get("scores", {}).get("reasons")]
    if not issues:
        issues = [["No major issues found", ""]]
    table = Table([["Page", "Issues"]] + issues[:30], colWidths=[3.5*inch, 3.5*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), primary),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey)
    ]))
    elems.append(table)
    elems.append(Spacer(1, 0.25*inch))
    # Add sample of off-page metrics (if available)
    offpage_summary = []
    for url,p in pages.items():
        if p.get("offpage") and (p["offpage"].get("domain_authority") or p["offpage"].get("page_authority")):
            offpage_summary.append((url, p["offpage"].get("domain_authority"), p["offpage"].get("page_authority")))
    if offpage_summary:
        elems.append(Paragraph("Off-page snapshot (some pages):", heading_style))
        for u, da, pa in offpage_summary[:10]:
            elems.append(Paragraph(f"{u} — DA: {da} — PA: {pa}", normal))
            elems.append(Spacer(1, 0.05*inch))
    elems.append(Spacer(1, 0.5*inch))
    elems.append(Paragraph("<font color='#A020F0'><b>VirtuNova — Where Creativity, Technology, and Strategy Converge.</b></font>", styles['Italic']))
    try:
        doc.build(elems)
        print(f"✅ PDF generated: {output_path}")
    except Exception as e:
        print(f"PDF generation error: {e}")

# ---------- Main runner ----------
async def run_audit(seed_url, output_path=None, max_pages=50, pagespeed_key=None, moz_access_id=None, moz_secret=None, write_web_ui=False):
    if not seed_url:
        raise ValueError("Please provide a URL")
    crawler = PlaywrightCrawler(seed_url, max_pages=max_pages)
    await crawler.run()
    report = {"site": seed_url, "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), "pages": {}}
    for url, data in crawler.results.items():
        page = {"fetch": data.get("fetch"), "analysis": data.get("analysis")}
        if page.get("analysis"):
            page["scores"] = score_page(page["analysis"], page.get("fetch", {}))
        # off-page metrics per page if keys provided
        if moz_access_id and moz_secret:
            page["offpage"] = fetch_moz_metrics(url, moz_access_id, moz_secret)
        report["pages"][url] = page
    report["canonical_chains"] = canonical_chain_check(report["pages"])
    report["json_ld_issues"] = {u: validate_json_ld(p["analysis"].get("json_ld", [])) for u,p in report["pages"].items() if p.get("analysis")}
    # pagespeed insights (site + first 3 pages)
    if pagespeed_key:
        candidates = [seed_url] + list(report["pages"].keys())[:3]
        tasks = [pagespeed_insights(p, pagespeed_key, strategy="mobile") for p in candidates]
        try:
            results = await asyncio.gather(*tasks)
            report["pagespeed"] = {p: r for p, r in zip(candidates, results)}
        except Exception as e:
            report["pagespeed_error"] = str(e)
    # write output JSON
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    # write CSV
    csv_path = os.path.join(os.path.dirname(output_path) or ".", "report.csv")
    write_csv_report(report, csv_path)
    # write web_ui
    if write_web_ui:
        os.makedirs("web_ui", exist_ok=True)
        with open(os.path.join("web_ui", "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    # generate PDF (summary)
    pdf_out = os.path.join(os.path.dirname(output_path) or ".", "SEO_Audit_Report.pdf")
    generate_pdf_report(report, output_path=pdf_out)
    return report

def cli():
    parser = argparse.ArgumentParser(description="Run full VirtuNova SEO audit")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", default="reports/report.json")
    parser.add_argument("--pages", type=int, default=50)
    parser.add_argument("--pagespeed-key", default=None)
    parser.add_argument("--moz-access-id", default=None)
    parser.add_argument("--moz-secret", default=None)
    parser.add_argument("--web-ui", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(run_audit(args.url, output_path=args.output, max_pages=args.pages, pagespeed_key=args.pagespeed_key, moz_access_id=args.moz_access_id, moz_secret=args.moz_secret, write_web_ui=args.web_ui))
    print("Audit complete. Output:", args.output)

if __name__ == "__main__":
    cli()
