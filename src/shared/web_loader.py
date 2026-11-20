"""
Reads websites and turns them into chunks, the same way chunker.py does for PDFs.

The links are listed in data/websites.txt, one per line, with the company first:
    azuria https://www.example.com/            <- home page: reads EVERY page of the site
    azuria https://www.example.com/faq         <- a deeper page: reads only that page
The company word must match the start of that company's PDF name ("azuria",
"wayfarra"), so the "only search this company" filter keeps working.

Finding every page of a site: first the site's sitemap (the page list it publishes
for search engines), else by following the links between its own pages.
Links are only used to find pages; other websites (LinkedIn etc.) are never opened,
and no links end up in the text.

Pages are saved in data/web_cache/. A saved copy younger than CACHE_HOURS is reused,
so restarts stay fast; older pages are downloaded again. If a download fails, the
old copy is used (the app still starts without internet).
"""
import hashlib
import html as htmllib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import lxml.html
import trafilatura
from trafilatura.sitemaps import sitemap_search
from langchain_core.documents import Document

from chunker import _split_documents, _prefix_source_file, table_to_chunks

MIN_TEXT_CHARS = 200  # less text than this: the page probably needs JavaScript to show its text
MAX_PAGES = 50        # most pages read from one site
CACHE_HOURS = 24      # saved pages younger than this are not downloaded again
DOWNLOADS_AT_ONCE = 8

# Links to these are files, not pages
NOT_A_PAGE = re.compile(r"\.(png|jpe?g|gif|svg|webp|ico|pdf|zip|css|js|json|xml|webmanifest|mp4|mp3)$", re.I)


def read_link_list(list_file):
    """[(company or None, url), ...] from websites.txt. Blank lines and # comments are skipped."""
    links = []
    for line in Path(list_file).read_text(encoding="utf-8").splitlines():
        # '#' starts a comment, but only at the start or after a space ('page#faq' is part of a link)
        line = re.sub(r"(^|\s)#.*$", "", line).strip()
        if not line:
            continue
        parts = line.split()
        if parts[0].startswith("http"):
            print(f"WARNING: no company given for {parts[0]}. Write it first, e.g. 'azuria {parts[0]}'")
            links.append((None, parts[0]))
        elif len(parts) >= 2 and parts[1].startswith("http"):
            links.append((parts[0].lower(), parts[1]))
        else:
            print(f"WARNING: could not read this line in {list_file}: '{line}'")
    return links


# ---------- saving and downloading ----------

def _saved(url, cache_dir, ending=".html"):
    return Path(cache_dir) / (hashlib.sha1(url.encode()).hexdigest()[:16] + ending)


def _is_fresh(path):
    return path.exists() and time.time() - path.stat().st_mtime < CACHE_HOURS * 3600


def fetch_page(url, cache_dir, fresh=False):
    """The page's HTML: a recent saved copy, else downloaded, else an old saved copy.
    fresh=True: always try to download first (used when checking the site for changes)."""
    saved = _saved(url, cache_dir)
    if not fresh and _is_fresh(saved):
        return saved.read_text(encoding="utf-8")
    html = trafilatura.fetch_url(url)
    if html:
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(html, encoding="utf-8")
        return html
    if saved.exists():
        print(f"WARNING: could not download {url}, using the copy saved earlier")
        return saved.read_text(encoding="utf-8")
    print(f"WARNING: could not download {url} and there is no saved copy. Skipping it.")
    return None


# ---------- finding all pages of a site ----------

def _site(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


def _normal(url):
    """Same page, one spelling: no '#part', no trailing slash."""
    url = urldefrag(url)[0]
    return url.rstrip("/") if urlparse(url).path not in ("", "/") else url.rstrip("/") + "/"


def is_home_page(url):
    return urlparse(url).path in ("", "/")


def _page_links(html, base_url):
    """Links in the page that go to other pages of the same site."""
    found = []
    for href in re.findall(r'href=["\']([^"\']+)["\']', html):
        url = _normal(urljoin(base_url, href.strip()))
        parsed = urlparse(url)
        if (parsed.scheme in ("http", "https") and _site(url) == _site(base_url)
                and not parsed.query and not NOT_A_PAGE.search(parsed.path)):
            found.append(url)
    return found


def _follow_links(start_url, cache_dir):
    """Finds pages by following the site's own links, starting at the home page."""
    to_visit, seen = [start_url], {start_url}
    while to_visit and len(seen) < MAX_PAGES:
        html = fetch_page(to_visit.pop(0), cache_dir)
        for url in _page_links(html or "", start_url):
            if url not in seen and len(seen) < MAX_PAGES:
                seen.add(url)
                to_visit.append(url)
    return list(seen)


def find_all_pages(home_url, cache_dir, fresh=False):
    """Every page of the site (at most MAX_PAGES), home page first.
    The list is saved too, so a restart without internet still reads every page."""
    home = _normal(home_url)
    saved_list = _saved(home, cache_dir, ".pages.txt")
    if not fresh and _is_fresh(saved_list):
        return saved_list.read_text(encoding="utf-8").split()

    try:
        pages = [_normal(u) for u in sitemap_search(home) if _site(u) == _site(home)]
    except Exception as error:
        print(f"Sitemap search failed for {home}: {error}")
        pages = []
    how = "sitemap"
    # OLD: without the sitemap it always followed links, which finds fewer pages
    # if not pages:
    #     pages = _follow_links(home, cache_dir)
    #     how = "following the site's links"
    if not pages and saved_list.exists():
        # Sitemap not reachable (e.g. the server is blocked): use the full page list saved earlier
        pages = saved_list.read_text(encoding="utf-8").split()
        how = "page list saved earlier (sitemap not reachable)"
    elif not pages:
        pages = _follow_links(home, cache_dir)
        how = "following the site's links"
    pages = list(dict.fromkeys([home] + pages))  # home page first, no doubles
    pages = [u for u in pages if not NOT_A_PAGE.search(urlparse(u).path)][:MAX_PAGES]

    if len(pages) > 1:
        saved_list.parent.mkdir(parents=True, exist_ok=True)
        saved_list.write_text("\n".join(pages), encoding="utf-8")
    elif saved_list.exists():  # nothing found now (no internet?): use the old list
        pages = saved_list.read_text(encoding="utf-8").split()
    print(f"{home}: found {len(pages)} pages ({how})")
    return pages


# ---------- page text -> chunks ----------

def _clean(text):
    """Removes leftovers of the page's code: HTML tags, '\\[edit\\]' links, escape backslashes."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\\([\[\]*_#|()])", r"\1", text)
    text = re.sub(r"^\s*\[edit\]\s*$", "", text, flags=re.M)
    return text


def _cells(line):
    """'| a | b |' -> ['a', 'b']"""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def split_markdown(text):
    """Splits the page into prose and tables (like chunker.py does for PDF pages).
    Returns (prose_text, [(title, rows), ...]). A table's title is the last heading
    or question above it, so every table piece still says what it is about."""
    prose, tables, rows = [], [], []
    title = None
    for line in text.splitlines() + [""]:
        stripped = line.strip()
        if stripped.startswith("|"):
            if not re.fullmatch(r"\|?[\s:\-|]+\|?", stripped):  # skip the '|---|---|' line
                rows.append(_cells(stripped))
            continue
        if rows:
            tables.append((title, rows))
            rows = []
        heading = re.match(r"^#{1,6}\s+(.*)", stripped)
        if heading:
            stripped = heading.group(1).strip()
            title = stripped
        elif stripped.endswith("?"):
            title = stripped
        prose.append(stripped)
    prose_text = re.sub(r"\n{3,}", "\n\n", "\n".join(prose)).strip()
    return prose_text, tables


def _drop_repeats(prose, seen_lines, min_chars=20):
    """Removes lines already read on another page of the same site (the same
    banner or intro on every page would otherwise be stored many times).
    Lines shorter than min_chars are kept even when repeated."""
    kept = []
    for line in prose.splitlines():
        key = " ".join(line.lower().split())
        if key and len(key) >= min_chars and key in seen_lines:
            continue
        seen_lines.add(key)
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def source_name(company, url):
    """'azuria', 'https://www.azuria.ae/faq' -> 'Azuria website (azuria.ae/faq)'.
    It starts with the company, so the company filter in rag_retriever.py finds it."""
    parsed = urlparse(url)
    where = (parsed.netloc.removeprefix("www.") + parsed.path).rstrip("/")
    return f"{company.capitalize()} website ({where})" if company else f"Website ({where})"


def _strings_in(data):
    """Every text value inside a JSON object, however deep."""
    if isinstance(data, dict):
        for value in data.values():
            yield from _strings_in(value)
    elif isinstance(data, list):
        for value in data:
            yield from _strings_in(value)
    elif isinstance(data, str):
        yield data


DRAFT_STYLE = {"header-one": "# ", "header-two": "## ", "header-three": "### ", "header-four": "#### ",
               "header-five": "##### ", "header-six": "###### ",
               "unordered-list-item": "- ", "ordered-list-item": "1. "}


def script_text(html):
    """Article text that the page only shows by running its scripts.
    Some site builders (GoDaddy blogs and other 'Draft.js' editors) put the article
    inside the page as data ({"blocks": [{"text": ..., "type": ...}]}) instead of as
    normal text, so the normal extraction only finds the menu. Returns '' if none."""
    decoder = json.JSONDecoder()
    best = []
    # Data is stored as 'window.X = {...}' or in '<script type="application/json">{...}'
    starts = [m.end() for m in re.finditer(r"=\s*(?=\{)", html)]
    starts += [m.end() for m in re.finditer(r"<script[^>]*application/(?:ld\+)?json[^>]*>\s*(?=\{)", html)]
    for start in starts[:200]:
        try:
            data, _ = decoder.raw_decode(html, start)
        except ValueError:
            continue
        for value in _strings_in(data):
            if '"blocks"' not in value:
                continue
            try:
                blocks = json.loads(value).get("blocks", [])
            except (ValueError, AttributeError):
                continue
            if len(blocks) > len(best):
                best = blocks
    lines = [DRAFT_STYLE.get(b.get("type"), "") + b.get("text", "").strip()
             for b in best if isinstance(b, dict) and b.get("text", "").strip()]
    return "\n\n".join(lines)


def page_title(html):
    found = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    return htmllib.unescape(found.group(1)).strip() if found else ""


# READ_ALL_TEXT = True: keep ALL the text a visitor can see on the page (menus, slideshows,
# short captions...), except the site menu (<nav>). A line repeated on the same page is kept once;
# a long line (20+ characters, like the footer or cookie notice) repeated on other pages is kept once per site.
# False: the old way, trafilatura guesses the "main content" and drops what looks like
# decoration; it dropped real content (Tiny Tales on the AI Immersion Lab page, the home page slideshow).
READ_ALL_TEXT = True

# Never text a visitor reads, or (nav) the site menu, the same on every page
_NOT_TEXT = {"script", "style", "noscript", "svg", "template", "head", "iframe", "select", "option", "nav"}
# These start a new line
_BLOCKS = {"p", "div", "section", "article", "header", "footer", "nav", "main", "aside", "li", "ul",
           "ol", "br", "tr", "td", "th", "table", "span", "a", "button", "label",
           "h1", "h2", "h3", "h4", "h5", "h6"}


def all_visible_text(html):
    """Every piece of text on the page, one line per block, headings marked with '#'.
    Repeats on this page are left out (GoDaddy repeats slideshow titles and labels): a long line
    (20+ characters) anywhere, a short one only right after itself, so short headings like
    "AI Solutions" stay even though the menu has the same words."""
    lines, words, seen = [], [], set()

    def end_line(prefix=""):
        line = " ".join("".join(words).split())
        words.clear()
        key = line.lower()
        if not line or (len(key) >= 20 and key in seen) or (lines and lines[-1].lstrip("# ").lower() == key):
            return
        seen.add(key)
        lines.append(prefix + line)

    def walk(element):
        tag = element.tag if isinstance(element.tag, str) else ""  # comments have no tag name
        if tag not in _NOT_TEXT:
            heading = "#" * int(tag[1]) + " " if re.fullmatch(r"h[1-6]", tag) else ""
            if tag in _BLOCKS:
                end_line()
            words.append(element.text or "")
            for child in element:
                walk(child)
            if tag in _BLOCKS:
                end_line(heading)
        words.append(element.tail or "")

    walk(lxml.html.fromstring(html))
    end_line()
    return "\n\n".join(lines)


def page_to_chunks(company, url, html, chunk_size, chunk_overlap, max_table_chars, seen_lines):
    if READ_ALL_TEXT:
        text = all_visible_text(html)
    else:
        text = trafilatura.extract(html, url=url, output_format="markdown", include_tables=True,
                                   include_links=False, include_comments=False, include_images=False)
    hidden = script_text(html)
    if len(hidden) > len(text or ""):
        # The real article is in the page's data; the normal text was only the menu
        title = page_title(html)
        text = (f"# {title}\n\n" if title else "") + hidden
    if not text:
        print(f"WARNING: no text found on {url} (it may need JavaScript to show its text)")
        return []

    name = source_name(company, url)
    prose, tables = split_markdown(_clean(text))
    prose = _drop_repeats(prose, seen_lines)
    meta = {"source_file": name, "page": 1, "url": url}

    prose_docs = [Document(page_content=prose, metadata={**meta, "type": "text"})] if prose else []
    chunks = _split_documents(prose_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    table_docs = []
    for t_idx, (title, rows) in enumerate(tables):
        for c_idx, piece in enumerate(table_to_chunks(rows, max_table_chars, title)):
            table_docs.append(Document(page_content=piece, metadata={
                **meta, "type": "table", "table_index": t_idx, "part": c_idx}))
    return chunks + _prefix_source_file(table_docs)


def load_websites(list_file, chunk_size, chunk_overlap, cache_dir="data/web_cache", max_table_chars=1500,
                  fresh=False):
    """Chunks for every link in list_file, ready for the vector store (same format as the PDF chunks).
    A home-page link reads the whole site; any other link reads just that page.
    fresh=True: download every page again, even if a recent saved copy exists."""
    if not Path(list_file).exists():
        print(f"No website list at {list_file}, skipping websites")
        return []
    links = read_link_list(list_file)
    print(f"Reading {len(links)} website link(s) ...\n")

    chunks = []
    for company, url in links:
        pages = find_all_pages(url, cache_dir, fresh) if is_home_page(url) else [url]

        # Download several pages at the same time (much faster than one by one)
        with ThreadPoolExecutor(DOWNLOADS_AT_ONCE) as pool:
            htmls = list(pool.map(lambda u: fetch_page(u, cache_dir, fresh), pages))

        seen_lines = set()
        site_chunks, short_pages = [], 0
        for page_url, html in zip(pages, htmls):
            if not html:
                continue
            page_chunks = page_to_chunks(company, page_url, html, chunk_size, chunk_overlap,
                                         max_table_chars, seen_lines)
            if sum(len(c.page_content) for c in page_chunks) < MIN_TEXT_CHARS:
                short_pages += 1
            site_chunks.extend(page_chunks)

        read = sum(1 for h in htmls if h)
        print(f"{url}: read {read} of {len(pages)} pages, {len(site_chunks)} chunks"
              + (f" ({short_pages} pages had almost no text)" if short_pages else ""))
        chunks.extend(site_chunks)

    print(f"Websites gave {len(chunks)} chunks\n")
    return chunks

