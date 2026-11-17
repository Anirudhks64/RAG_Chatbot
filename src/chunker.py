from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
import os
import re
from collections import Counter
from langchain_core.documents import Document
import statistics

# Function to load pdf files as chunks 
def load_pdf_files(pdf_files, chunk_size, chunk_overlap):
    
    chunks = []
    for doc in pdf_files: 
        prose_docs, table_chunks = _load_pdf(doc)
        prose_chunks = _split_documents(prose_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        table_chunks = _prefix_source_file(table_chunks)
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
            found = page.find_tables()
            tables = [t for t in found if not _is_page_frame(t, found, page)]
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

def _is_page_frame(table, all_tables, page):
    """pdfplumber sometimes reads a border around the page as one huge 'table'
    (the last page of both PDFs). Treating it as a table would throw away all
    the text on that page, so a 'table' that covers most of the page, or that
    has other tables inside it, is skipped."""
    x0, top, x1, bottom = table.bbox
    covers_page = (x1 - x0) * (bottom - top) > 0.6 * page.width * page.height

    def is_inside_this(box):
        return _inside({"x0": box[0], "top": box[1], "x1": box[2], "bottom": box[3]}, table.bbox)

    has_tables_inside = any(is_inside_this(other.bbox) for other in all_tables if other is not table)
    return covers_page or has_tables_inside


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
def _split_documents(documents, chunk_size, chunk_overlap):
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators = ["\n\n", "\n", " ", ""],
        add_start_index = True,  # where each chunk starts on its page (used below)
    )


    split_docs = text_splitter.split_documents(documents)
    split_docs = _add_question_titles(split_docs, documents)
    split_docs = _prefix_source_file(split_docs=split_docs)

    return split_docs


PAGE_NUMBER = re.compile(r"^Page \d+ of \d+$")


def _is_question(line):
    return line.strip().endswith("?")


def _is_section(line):
    """'3. Flights' is a section heading; '2. We register the unit ... register.' is a list step."""
    line = line.strip()
    return bool(re.match(r"^\d+\.\s+\S", line)) and len(line.split()) <= 7 and not line.endswith(".")


def _add_question_titles(chunks, pages):
    """A chunk that starts in the middle of an FAQ answer gets that FAQ's question
    as a title ("Title: Are there any extra charges for the Maldives package?").
    Without it, an answer cut off from its question (by a chunk or page break)
    is almost never found, because nothing in it says what it is about."""
    # Lines repeated on many pages are page headers, not content
    seen_on = Counter(line.strip() for page in pages for line in set(page.page_content.splitlines()))
    repeated = {line for line, count in seen_on.items() if count >= 3}

    def content_lines(text):
        return [l for l in text.splitlines()
                if l.strip() and l.strip() not in repeated and not PAGE_NUMBER.match(l.strip())]

    # Every question line with its position, and the question still open when each page starts
    questions, open_at_start, last = {}, {}, {}
    for page in pages:
        key = (page.metadata["source_file"], page.metadata["page"])
        open_at_start[key] = last.get(page.metadata["source_file"])
        offset, found = 0, []
        for line in page.page_content.splitlines(keepends=True):
            if _is_question(line) and line.strip() not in repeated:
                found.append((offset, line.strip()))
            offset += len(line)
        questions[key] = found
        if found:
            last[page.metadata["source_file"]] = found[-1][1]

    for chunk in chunks:
        lines = content_lines(chunk.page_content)
        if not lines or _is_question(lines[0]) or _is_section(lines[0]):
            continue  # the chunk starts with its own question (or a new section)
        key = (chunk.metadata["source_file"], chunk.metadata["page"])
        start = chunk.metadata.get("start_index", 0)
        before = [q for pos, q in questions.get(key, []) if pos < start]
        question = before[-1] if before else open_at_start.get(key)
        if question:
            chunk.page_content = f"Title: {question}\n{chunk.page_content}"
    return chunks


# Prefixes Document name to page.content so database can access source file name during retrieval
def _prefix_source_file(split_docs): 
    for chunk in split_docs: 
        prefix = chunk.metadata['source_file'].removesuffix(".pdf") + ": "
        if not chunk.page_content.startswith(prefix):
            chunk.page_content = prefix + chunk.page_content
    return split_docs







