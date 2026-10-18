from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
import os
from langchain_core.documents import Document
import statistics

# Function to load pdf files as chunks 
def load_pdf_files(pdf_files, chunk_size, chunk_overlap):
    
    chunks = []
    for doc in pdf_files: 
        prose_docs, table_chunks = _load_pdf(doc)
        prose_chunks = split_documents(prose_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks.extend(prose_chunks + table_chunks)

    print(f'split {len(pdf_files)} documents into {len(chunks)} chunks')
    print("\n")
    return chunks


# Loads PDF Files in a list separating Prose Content and Table Contents and returns Langchain Documents
def _load_pdf(path, max_table_chars=1500):
    source_file = os.path.basename(path)
    prose_docs, table_docs = [], []
    current_title = None

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            bboxes = [t.bbox for t in tables]
            headings = _page_headings(page, bboxes)      

            # prose = everything on the page that is NOT inside a table
            prose_page = page.filter(
                lambda o: not any(_inside(o, b) for b in bboxes)
            )
            text = prose_page.extract_text() or ""
            if text.strip():
                prose_docs.append(Document(
                    page_content=text,
                    metadata={"source_file": source_file, "page": page_num, "type": "text"},
                ))

            for t_idx, table in enumerate(tables):
                above = [h for y, h in headings if y < table.bbox[1]]
                title = above[-1] if above else current_title
                for c_idx, chunk in enumerate(table_to_chunks(table.extract(), max_table_chars, title)):
                    table_docs.append(Document(
                        page_content=chunk,
                        metadata={"source_file": source_file, "page": page_num,
                                  "type": "table", "table_index": t_idx, "part": c_idx},
                    ))
             
            if headings:
                current_title = headings[-1][1]

    return prose_docs, table_docs


def _page_lines(page):
    """Group the words on a page into lines, with each line's font size and boldness."""
    words = page.extract_words(extra_attrs=["size", "fontname"])
    words.sort(key=lambda w: (w["bottom"], w["x0"]))

    lines = []
    for w in words:
        if lines and abs(w["bottom"] - lines[-1]["bottom"]) <= 3:
            lines[-1]["words"].append(w)
        else:
            lines.append({"bottom": w["bottom"], "words": [w]})

    result = []
    for l in lines:
        ws = sorted(l["words"], key=lambda w: w["x0"])
        fonts = [w["fontname"].lower() for w in ws]
        result.append({
            "text": " ".join(w["text"] for w in ws),
            "size": max(w["size"] for w in ws),
            "bold": all(("bold" in f or "black" in f) for f in fonts),
            "top": min(w["top"] for w in ws),
            "bottom": max(w["bottom"] for w in ws),
            "fontname": ws[0]["fontname"],
        })
    return result


def _page_headings(page, table_bboxes=(), ignore_top=0):
    """ Return [(y_position, heading_text), ...] for every heading-like line on the page """
    lines = _page_lines(page)
    if not lines:
        return []
    body_size = statistics.median(l["size"] for l in lines)

    headings = []
    for l in lines:
        if l["top"] <= ignore_top:
            continue
        # skip lines that are inside a table (table header cells are often bold)
        if any(l["top"] >= b[1] - 2 and l["bottom"] <= b[3] + 2 for b in table_bboxes):
            continue

        is_bigger = l["size"] > body_size * 1.08
        is_bold_short = l["bold"] and len(l["text"].split()) <= 12
        if not (is_bigger or is_bold_short):
            continue

        # a heading that wraps onto 2 lines: same size, right under the previous one
        if headings:
            prev = headings[-1]
            if abs(prev["size"] - l["size"]) < 0.5 and 0 <= l["top"] - prev["bottom"] < prev["size"] * 0.8:
                prev["text"] += " " + l["text"]
                prev["bottom"] = l["bottom"]
                continue
        headings.append(dict(l))

    return [(h["top"], h["text"]) for h in headings]


def debug_lines(page):
    """Print every line with its font size and boldness, to see what the PDF really contains."""
    for l in _page_lines(page):
        print(f'{l["top"]:6.0f}  size={l["size"]:5.1f}  bold={l["bold"]!s:5}  {l["text"][:60]}')


def _inside(obj, bbox, pad=2):
    x0, top, x1, bottom = bbox
    return (obj["x0"] >= x0 - pad and obj["x1"] <= x1 + pad and
            obj["top"] >= top - pad and obj["bottom"] <= bottom + pad)


def _clean(cell):
    return " ".join((cell or "").split())


def _row_md(row):
    return "| " + " | ".join(_clean(c) for c in row) + " |"


def table_to_chunks(rows, max_chars, title=None):
    """Split a table by rows, repeating the title and header in every chunk."""
    rows = [r for r in rows if any(_clean(c) for c in r)]
    if len(rows) < 2:
        return []
    header = _row_md(rows[0])
    divider = "| " + " | ".join("---" for _ in rows[0]) + " |"
    title_line = f"Title: {title}\n\n" if title else ""
    head = f"{title_line}{header}\n{divider}\n"

    chunks, current = [], head
    for row in rows[1:]:
        line = _row_md(row) + "\n"
        if len(current) + len(line) > max_chars and current != head:
            chunks.append(current.rstrip())
            current = head
        current += line
    if current != head:
        chunks.append(current.rstrip())
    return chunks

### Text splitting into chunks 
### Returns list of Document data types which are the chunks 
def split_documents(documents, chunk_size, chunk_overlap):
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap, 
        length_function = len, 
        separators = ["\n\n", "\n", " ", ""]
    )


    split_docs = text_splitter.split_documents(documents)
    split_docs = _prefix_source_file(split_docs=split_docs)    
    
    return split_docs


# Prefixes Document name to page.content so database can access source file name during retrieval
def _prefix_source_file(split_docs): 
    for chunk in split_docs: 
        prefix = chunk.metadata['source_file'].removesuffix(".pdf") + ": "
        if not chunk.page_content.startswith(prefix):
            chunk.page_content = prefix + chunk.page_content
    return split_docs







