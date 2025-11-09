#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VirtuNova SEO Audit Toolkit — Hybrid (Playwright + PageSpeed)
- Uses Playwright (Chromium) to render pages (works for JS-heavy sites)
- Uses Google PageSpeed Insights (optional) for performance/SEO metrics
- Produces: reports/report.json, reports/SEO_Audit_Report.pdf and web_ui/report.json
Usage:
  python seo_audit_tool_extended.py --url https://www.ellocentlabs.com --output reports/report.json --pages 50 --pagespeed-key $PAGESPEED_KEY --web-ui
Dependencies:
  pip install -r requirements.txt
  pip install playwright
  playwright install chromium
(When running in GitHub Actions add playwright install step)
"""

import argparse
import asyncio
import json
import os
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse

# Optional/required libs
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

# PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib import colors
    from reportlab.lib.units import inch
except Exception:
    # PDF optional — will error if not installed
    SimpleDocTemplate = None

# Constants
USER_AGENT = "VirtuNova-SEO-Toolkit/1.0 (+https://virtunova.com)"
REQUEST_TIMEOUT = 30
MAX_CONCURRENT_BROWSERS = 4
DEFAULT_MAX_PAGES = 100

# -------------------------
# Utilities
# -------------------------
def normalize_url(base, href):
    if not href: 
        return None
    href = href.strip()
    if href.startswith(("javascript:", "mailto:", "#")):
        return None
    joined = urljoin(base, href)
    parsed = urlparse(joined)
    cleaned = parsed._replace(fragment='')
    return cleaned.geturl()

def same_origin(a, b):
    pa, pb = urlparse(a), urlparse(b)
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)

def safe_get_text(soup, selector, attr=None):
    try:
        el = soup.select_one(selector)
        if not el:
            return ""
        if attr:
            return el.get(attr, "").strip()
        return el.get_text(strip=True)
    except Exception:
        return ""

# -------------------------
# Playwright renderer
# -------------------------
async def render_page_content(url, timeout=30000):
    """
    Render the page with Playwright and return:
      { "url": url, "status": status, "content": html, "title": title, "headers": headers }
    """
    if not async_playwright:
        return {"url": url, "status": None, "error": "playwright-not-installed"}
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"], headless=True)
            page = await browser.new_page(user_agent=USER_AGENT)
            try:
                resp = await page.goto(url, timeout=timeout, wait_until="networkidle")
            except PWTimeout:
                # Try again with less strict wait
                try:
                    resp = await page.goto(url, timeout=timeout, wait_until="load")
                except Exception as e:
                    await browser.close()
                    return {"url": url, "status": None, "error": f"navigation-timeout: {e}"}
            status = resp.status if resp else None
            # wait a short time for dynamic content to settle
            try:
                await page.wait_for_timeout(500)
            except Exception:
                pass
            content = await page.content()
            title = await page.title()
            headers = {}
            try:
                if resp:
                    # resp.headers() is dict-like
                    headers = dict(await resp.all_headers())
            except Exception:
                headers = {}
            await browser.close()
            return {"url": url, "status": status, "content": content, "title": title, "headers": headers}
    except Exception as e:
        return {"url": url, "status": None, "error": str(e)}

# -------------------------
# Analysis helpers
# -------------------------
def analyze_html_from_text(url, html):
    if not BeautifulSoup or not html:
        # minimal fallback
        return {"title": "", "meta_description": "", "h1": [], "images": [], "links": [], "word_count": 0}
    soup = BeautifulSoup(html, "lxml")
    title = safe_get_text(soup, "title")
    meta_desc = ""
    desc_tag = soup.find("meta", attrs={"name": re.compile("description", re.I)})
    if desc_tag and desc_tag.get("content"):
        meta_desc = desc_tag.get("content").strip()
    h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]
    imgs = [img.get("src") for img in soup.find_all("img")]
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    body_text = soup.body.get_text(" ", strip=True) if soup.body else ""
    words = re.findall(r"\w+", body_text)
    hreflangs = []
    for l in soup.find_all("link", attrs={"rel": re.compile("alternate", re.I)}):
        if l.get("hreflang") and l.get("href"):
            hreflangs.append({"hreflang": l.get("hreflang"), "href": l.get("href")})
    # JSON-LD blocks
    json_ld = []
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            if s.string:
                json_ld.append(json.loads(s.string))
        except Exception:
            json_ld.append({"raw": (s.string or "")[:500]})
    return {
        "title": {"text": title, "length": len(title)},
        "meta_description": {"text": meta_desc, "length": len(meta_desc)},
        "h1": {"count": len(h1s), "texts": h1s},
        "images": {"total": len(imgs), "missing_alt_count": sum(1 for img in soup.find_all("img") if not img.get("alt"))},
        "links": {"count": len(links)},
        "word_count": len(words),
        "hreflangs": hreflangs,
        "json_ld": json_ld
    }

def validate_json_ld(json_ld_blocks):
    issues = []
    for i,b in enumerate(json_ld_blocks):
        if isinstance(b, dict):
            if '@context' not in b and '@graph' not in b:
                issues.append({'index': i, 'issue': 'missing @context'})
            if '@type' not in b and '@graph' not in b:
                issues.append({'index': i, 'issue': 'missing @type'})
        else:
            issues.append({'index': i, 'issue': 'not a dict'})
    return issues

def canonical_chain_check(pages):
    chains = []
    for url, data in pages.items():
        c = data.get('analysis', {}).get('canonical')
        if not c:
            continue
        chain = [url]
        nxt = c
        while nxt and nxt != chain[-1] and nxt in pages and len(chain) < 20:
            chain.append(nxt)
            nxt = pages[nxt].get('analysis', {}).get('canonical')
        if len(chain) > 1:
            chains.append(chain)
    return chains

# -------------------------
# PageSpeed Insights
# -------------------------
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

# -------------------------
# Crawler (Playwright-based)
# -------------------------
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
            raise RuntimeError("Playwright is required for the hybrid crawler. Install 'playwright' and run 'playwright install chromium'.")
        # Use a pool of workers by launching multiple contexts sequentially
        sem = asyncio.Semaphore(MAX_CONCURRENT_BROWSERS)
        async def worker():
            while self.to_visit:
                url = self.to_visit.popleft()
                async with sem:
                    try:
                        page_result = await render_page_content(url)
                    except Exception as e:
                        page_result = {"url": url, "status": None, "error": str(e)}
                    self.results[url] = {"fetch": page_result}
                    if page_result.get("status") and page_result.get("content"):
                        analysis = analyze_html_from_text(url, page_result["content"])
                        self.results[url]["analysis"] = analysis
                        # extract links via BeautifulSoup
                        try:
                            from bs4 import BeautifulSoup as _BS
                            soup = _BS(page_result["content"], "lxml")
                            for a in soup.find_all("a", href=True):
                                n = normalize_url(url, a["href"])
                                if not n:
                                    continue
                                # stay within same origin
                                if urlparse(n).netloc == self.seed_netloc and n not in self.seen and len(self.seen) < self.max_pages:
                                    self.seen.add(n)
                                    self.to_visit.append(n)
                        except Exception:
                            pass
        # Run a small number of concurrent workers
        tasks = [asyncio.create_task(worker()) for _ in range(max(1, min(6, MAX_CONCURRENT_BROWSERS)))]
        await asyncio.gather(*tasks)

# -------------------------
# Scoring and report generation
# -------------------------
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

def scaffold_web_ui(report, target_dir="web_ui"):
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

def generate_pdf_report(report_data, output_path="reports/SEO_Audit_Report.pdf"):
    if SimpleDocTemplate is None:
        print("PDF generation skipped — reportlab not installed.")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
    elems.append(Paragraph("VirtuNova — SEO Audit Report", title_style))
    elems.append(Spacer(1, 0.15*inch))
    elems.append(Paragraph(f"<b>Website:</b> {report_data.get('site','')}", normal))
    elems.append(Paragraph(f"<b>Generated:</b> {report_data.get('generated_at','')}", normal))
    elems.append(Spacer(1, 0.2*inch))
    pages = report_data.get("pages", {})
    total_pages = len(pages)
    elems.append(Paragraph("Executive Summary", heading_style))
    elems.append(Paragraph(f"Total pages crawled: {total_pages}", normal))
    scores = [p.get("scores", {}).get("score") for p in pages.values() if p.get("scores")]
    avg = round(sum(scores)/max(1,len(scores)),1) if scores else 0
    elems.append(Paragraph(f"Average SEO Score: <b>{avg}%</b>", normal))
    elems.append(Spacer(1, 0.2*inch))
    # Top issues
    issues = [[url, ", ".join(p.get("scores", {}).get("reasons", []))] for url,p in pages.items() if p.get("scores", {}).get("reasons")]
    if not issues:
        issues = [["No major issues found", ""]]
    table = Table([["Page URL", "Detected Issues"]] + issues[:20], colWidths=[3.0*inch, 3.0*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), primary),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey)
    ]))
    elems.append(table)
    elems.append(Spacer(1,0.25*inch))
    elems.append(Paragraph("<b><font color='#A020F0'>VirtuNova</font></b> — Where Creativity, Technology, and Strategy Converge.", ParagraphStyle("Footer", textColor=colors.grey, fontSize=10, alignment=1)))
    try:
        doc.build(elems)
        print(f"✅ PDF generated: {output_path}")
    except Exception as e:
        print(f"PDF generation error: {e}")

# -------------------------
# Main runner
# -------------------------
async def run_audit(seed_url, output_path=None, max_pages=50, pagespeed_key=None, write_web_ui=False):
    if not seed_url:
        raise ValueError("Please provide a URL")
    crawler = PlaywrightCrawler(seed_url, max_pages=max_pages)
    await crawler.run()
    report = {"site": seed_url, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "pages": {}}
    for url, data in crawler.results.items():
        page = {"fetch": data.get("fetch"), "analysis": data.get("analysis")}
        if page.get("analysis"):
            page["scores"] = score_page(page["analysis"], page.get("fetch", {}))
        report["pages"][url] = page
    report["canonical_chains"] = canonical_chain_check(report["pages"])
    report["json_ld_issues"] = {u: validate_json_ld(p["analysis"].get("json_ld", [])) for u,p in report["pages"].items() if p.get("analysis")}
    # PageSpeed (seed + first few pages)
    if pagespeed_key:
        candidates = [seed_url] + list(report["pages"].keys())[:3]
        tasks = [pagespeed_insights(p, pagespeed_key, strategy="mobile") for p in candidates]
        try:
            results = await asyncio.gather(*tasks)
            report["pagespeed"] = {p: r for p,r in zip(candidates, results)}
        except Exception as e:
            report["pagespeed_error"] = str(e)
    # lighthouse import not included here, could be added
    # write outputs
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    if write_web_ui:
        scaffold_web_ui(report, target_dir="web_ui")
    return report

def main():
    parser = argparse.ArgumentParser(description="VirtuNova Hybrid SEO Audit (Playwright + PageSpeed)")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", default="reports/report.json")
    parser.add_argument("--pages", type=int, default=50)
    parser.add_argument("--pagespeed-key", default=None)
    parser.add_argument("--web-ui", action="store_true")
    args = parser.parse_args()

    # Run audit
    print("🚀 Starting VirtuNova Hybrid SEO Audit for", args.url)
    report = asyncio.run(run_audit(args.url, output_path=args.output, max_pages=args.pages, pagespeed_key=args.pagespeed_key, write_web_ui=args.web_ui))
    print("✅ Audit finished. Output written to:", args.output)

    # Generate PDF only if we have pages
    try:
        if os.path.exists(args.output):
            with open(args.output, "r", encoding="utf-8") as f:
                r = json.load(f)
            if r.get("pages"):
                generate_pdf_report(r, output_path=os.path.join(os.path.dirname(args.output), "SEO_Audit_Report.pdf"))
            else:
                print("⚠️ No pages found in report.json — skipping PDF generation.")
    except Exception as e:
        print("❌ PDF generation failed:", e)

if __name__ == "__main__":
    main()
