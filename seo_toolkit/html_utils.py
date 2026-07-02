"""HTML parsing helpers — consolidates repeated meta/link tag extraction."""

import re


def extract_meta_content(soup, name):
    """Extract the ``content`` attribute from a <meta name="..."> tag.

    Performs a case-insensitive match on the *name* attribute.
    Returns the stripped content string, or "" if the tag is missing.
    """
    tag = soup.find("meta", attrs={"name": re.compile(name, re.I)})
    if tag and tag.get("content"):
        return tag["content"].strip()
    return ""


def extract_link_href(soup, rel, attr=None):
    """Extract the ``href`` from a <link rel="..."> tag.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed HTML document.
    rel : str
        Value to match against the ``rel`` attribute (case-insensitive regex).
    attr : str | None
        If given, only match tags that also have this attribute present.

    Returns
    -------
    str
        The stripped href value, or "" if not found.
    """
    kwargs = {"rel": re.compile(rel, re.I)}
    tag = soup.find("link", **kwargs)
    if attr:
        if tag and tag.get(attr):
            return tag[attr].strip()
    else:
        if tag and tag.get("href"):
            return tag["href"].strip()
    return ""
