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
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import socket
import time
from collections import deque
from datetime import datetime
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

from seo_toolkit.imports import (
    async_playwright,
    PWTimeout,
    aiohttp,
    BeautifulSoup,
    SimpleDocTemplate,
    A4,
    ParagraphStyle,
    getSampleStyleSheet,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    colors,
    inch,
)
from seo_toolkit.file_utils import ensure_parent_dir, write_json, write_csv_report
from seo_toolkit.html_utils import extract_meta_content, extract_link_href
from seo_toolkit.scoring import compute_page_score

# ---------- Config ----------
USER_AGENT = "VirtuNova-SEO-Toolkit/1.0 (+https://virtunova.com)"
DEFAULT_MAX_PAGES = 100
REQUEST_TIMEOUT = 30
PLAYWRIGHT_TIMEOUT_MS = 30000
ALLOWED_SCHEMES = {"http", "https"}
SENSITIVE_HEADERS = {
    "set-cookie", "authorization", "proxy-authorization",
    "cookie", "x-api-key", "x-csrf-token", "x-xsrf-token",
}

# ---------- Security helpers ----------
def validate_url(url):
    """Reject non-HTTP(S) schemes and private/internal IP targets."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Blocked URL scheme '{parsed.scheme}' — only http/https allowed"
        )
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname")
    try:
        resolved = socket.getaddrinfo(hostname, None)
        for _family, _type, _proto, _canonname, sockaddr in resolved:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise ValueError(
                    f"Blocked request to private/internal address {ip} "
                    f"(resolved from {hostname})"
                )
    except socket.gaierror:
        pass  # let the actual request fail with a proper network error
    return url


def sanitize_output_path(path):
    """Ensure the output path stays under the current working directory."""
    resolved = os.path.realpath(path)
    cwd = os.path.realpath(os.getcwd())
    if not resolved.startswith(cwd + os.sep) and resolved != cwd:
        raise ValueError(
            f"Output path '{path}' resolves outside the working directory"
        )
    return resolved


def filter_headers(headers):
    """Strip sensitive headers before persisting to reports."""
    if not headers:
        return {}
    return {
        k: v for k, v in headers.items()
        if k.lower() not in SENSITIVE_HEADERS
    }

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
            try:
                await page.wait_for_timeout(500)
            except Exception:
                logger.debug("Timeout waiting for content settle on %s", url)
            content = await page.content()
            title = ""
            try:
                title = await page.title()
            except Exception as e:
                logger.warning("Failed to retrieve page title for %s: %s", url, e)
                title = ""
            headers = {}
            try:
                if resp:
                    headers = dict(await resp.all_headers())
            except Exception as e:
                logger.warning("Failed to retrieve response headers for %s: %s", url, e)
                headers = {}
            await browser.close()
            return {"url": url, "status": status, "content": content, "title": title, "headers": filter_headers(headers)}
    except Exception as e:
        return {"url": url, "status": None, "error": str(e)}

# ---------- HTML analysis ----------
def analyze_html_from_text(url, html):
    if not BeautifulSoup or not html:
        return {}
    soup = BeautifulSoup(html, "lxml")
    result = {}
    # title
    title_tag = soup.find("title")
    title = title_tag.string.strip() if title_tag and title_tag.string else ""
    result["title"] = {"text": title, "length": len(title)}
    # meta tags via shared utility
    desc = extract_meta_content(soup, "description")
    result["meta_description"] = {"text": desc, "length": len(desc)}
    result["meta_robots"] = extract_meta_content(soup, "robots")
    result["viewport"] = extract_meta_content(soup, "viewport")
    # headings
    h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]
    result["h1"] = {"count": len(h1s), "texts": h1s}
    # canonical via shared utility
    result["canonical"] = extract_link_href(soup, "canonical")
    # json-ld
    json_ld = []
    for s in soup.find_all("script", type="application/ld+json"):
        if s.string:
            try:
                json_ld.append(json.loads(s.string))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Malformed JSON-LD block on %s: %s", url, e)
                json_ld.append({"raw": s.string[:500], "parse_error": str(e)})
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
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(api, params=params) as resp:
                data = await resp.json()
                if resp.status != 200:
                    logger.warning(
                        "PageSpeed API returned HTTP %d for %s: %s",
                        resp.status, url, data.get("error", {}).get("message", "unknown error"),
                    )
                return data
        except aiohttp.ClientError as e:
            logger.error("PageSpeed network error for %s: %s", url, e)
            return {"error": f"{type(e).__name__}: {e}"}
        except asyncio.TimeoutError:
            logger.error("PageSpeed request timed out for %s", url)
            return {"error": "Request timed out"}
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("PageSpeed returned non-JSON response for %s: %s", url, e)
            return {"error": f"{type(e).__name__}: {e}"}

# ---------- Moz (Off-page metrics) ----------
def create_moz_auth(access_id, secret):
    expires = int(time.time()) + 300
    string_to_sign = f"{access_id}\n{expires}"
    h = hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha256)
    signature = h.digest()
    signature_b64 = signature.hex()
    return expires, signature_b64

def fetch_moz_metrics(url, access_id, secret):
    if not access_id or not secret:
        return {"domain_authority": None, "page_authority": None, "external_links": None, "moz_rank": None}
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
                        try:
                            soup = BeautifulSoup(page_result["content"], "lxml")
                            for a in soup.find_all("a", href=True):
                                n = normalize_url(url, a["href"])
                                if not n: continue
                                parsed_n = urlparse(n)
                                if parsed_n.scheme not in ALLOWED_SCHEMES:
                                    continue
                                if parsed_n.netloc == self.seed_netloc and n not in self.seen and len(self.seen) < self.max_pages:
                                    self.seen.add(n); self.to_visit.append(n)
                        except Exception as e:
                            logger.warning("Link extraction failed for %s: %s", url, e)
        tasks = [asyncio.create_task(worker()) for _ in range(3)]
        await asyncio.gather(*tasks)

# ---------- PDF report ----------
def generate_pdf_report(report_data, output_path="reports/SEO_Audit_Report.pdf"):
    if SimpleDocTemplate is None:
        print("PDF generation skipped — reportlab not installed.")
        return
    ensure_parent_dir(output_path)
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
        except Exception as e:
            logger.warning("Failed to embed logo in PDF: %s", e)
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
        logger.info("PDF generated: %s", output_path)
    except Exception as e:
        logger.error("PDF generation failed for %s: %s", output_path, e)
        raise

# ---------- Main runner ----------
async def run_audit(seed_url, output_path=None, max_pages=50, pagespeed_key=None, moz_access_id=None, moz_secret=None, write_web_ui=False):
    if not seed_url:
        raise ValueError("Please provide a URL")
    validate_url(seed_url)
    if output_path:
        sanitize_output_path(output_path)
    crawler = PlaywrightCrawler(seed_url, max_pages=max_pages)
    await crawler.run()
    report = {"site": seed_url, "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), "pages": {}}
    for url, data in crawler.results.items():
        page = {"fetch": data.get("fetch"), "analysis": data.get("analysis")}
        if page.get("analysis"):
            page["scores"] = compute_page_score(page["analysis"], page.get("fetch", {}))
        if moz_access_id and moz_secret:
            page["offpage"] = fetch_moz_metrics(url, moz_access_id, moz_secret)
        report["pages"][url] = page
    report["canonical_chains"] = canonical_chain_check(report["pages"])
    report["json_ld_issues"] = {u: validate_json_ld(p["analysis"].get("json_ld", [])) for u,p in report["pages"].items() if p.get("analysis")}
    # pagespeed insights (site + first 3 pages)
    if pagespeed_key:
        candidates = [seed_url] + list(report["pages"].keys())[:3]
        psi_tasks = [pagespeed_insights(p, pagespeed_key, strategy="mobile") for p in candidates]
        results = await asyncio.gather(*psi_tasks, return_exceptions=True)
        report["pagespeed"] = {}
        for candidate_url, result in zip(candidates, results):
            if isinstance(result, Exception):
                logger.error("PageSpeed Insights failed for %s: %s", candidate_url, result)
                report["pagespeed"][candidate_url] = {"error": f"{type(result).__name__}: {result}"}
            else:
                report["pagespeed"][candidate_url] = result
    # write outputs using shared file utilities
    if output_path:
        write_json(report, output_path)
        out_dir = os.path.dirname(output_path) or "."
        csv_path = os.path.join(out_dir, "report.csv")
        write_csv_report(report, csv_path)
        # generate PDF (summary) alongside JSON
        pdf_out = os.path.join(out_dir, "SEO_Audit_Report.pdf")
        try:
            generate_pdf_report(report, output_path=pdf_out)
        except Exception:
            pass  # already logged inside generate_pdf_report
    if write_web_ui:
        write_json(report, os.path.join("web_ui", "report.json"))
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
    try:
        validate_url(args.url)
    except ValueError as e:
        parser.error(str(e))
    if args.output:
        try:
            sanitize_output_path(args.output)
        except ValueError as e:
            parser.error(str(e))
    report = asyncio.run(run_audit(args.url, output_path=args.output, max_pages=args.pages, pagespeed_key=args.pagespeed_key, moz_access_id=args.moz_access_id, moz_secret=args.moz_secret, write_web_ui=args.web_ui))
    print("Audit complete. Output:", args.output)

if __name__ == "__main__":
    cli()
