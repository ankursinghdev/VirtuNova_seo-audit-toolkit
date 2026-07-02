"""Centralized optional dependency loading.

All third-party libraries that may not be installed are imported here once.
Modules throughout the toolkit import from this file instead of repeating
try/except blocks.
"""


def _try_import(import_fn):
    try:
        return import_fn()
    except ImportError:
        return None


# --- Playwright ---
def _load_playwright():
    from playwright.async_api import async_playwright, TimeoutError as PWTimeout
    return async_playwright, PWTimeout

_pw = _try_import(_load_playwright)
async_playwright = _pw[0] if _pw else None
PWTimeout = _pw[1] if _pw else Exception

# --- aiohttp ---
aiohttp = _try_import(lambda: __import__("aiohttp"))

# --- BeautifulSoup ---
def _load_bs4():
    from bs4 import BeautifulSoup
    return BeautifulSoup

BeautifulSoup = _try_import(_load_bs4)

# --- ReportLab ---
def _load_reportlab():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    return (SimpleDocTemplate, A4, ParagraphStyle, getSampleStyleSheet,
            Paragraph, Spacer, Table, TableStyle, Image, colors, inch)

_rl = _try_import(_load_reportlab)
if _rl:
    (SimpleDocTemplate, A4, ParagraphStyle, getSampleStyleSheet,
     Paragraph, Spacer, Table, TableStyle, Image, colors, inch) = _rl
else:
    SimpleDocTemplate = A4 = ParagraphStyle = getSampleStyleSheet = None
    Paragraph = Spacer = Table = TableStyle = Image = colors = inch = None
