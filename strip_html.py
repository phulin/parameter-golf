#!/usr/bin/env python3
"""Strip headers, navs, footers, and class attributes from HTML papers."""

import glob
from bs4 import BeautifulSoup

for path in glob.glob("*.html"):
    with open(path) as f:
        soup = BeautifulSoup(f, "html.parser")

    for tag in soup.find_all(["head", "header", "nav", "footer"]):
        tag.decompose()

    KEEP = {"alttext", "alt", "title", "href"}
    for tag in soup.find_all(True):
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in KEEP}

    with open(path, "w") as f:
        f.write(str(soup))

    print(f"Stripped {path}")
