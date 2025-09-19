import os, re, json, uuid, tempfile
from pathlib import Path
from typing import Optional, List, Dict

import jieba, jieba.analyse
import pdfplumber
from docx import Document
from ebooklib import epub
from bs4 import BeautifulSoup

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Novel Analyzer API", version="1.2.0")

def _read_txt(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")

def _read_pdf(fp: Path) -> str:
    out = []
    with pdfplumber.open(str(fp)) as pdf:
        for page in pdf.pages:
            out.append(page.extract_text() or "")
    return "\n".join(out)

def _read_docx(fp: Path) -> str:
    doc = Document(str(fp))
    return "\n".join(p.text for p in doc.paragraphs)

def _read_epub(fp: Path) -> str:
    book = epub.read_epub(str(fp))
    texts = []
    for item in book.get_items():
        if item.get_type() == 9:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            texts.append(soup.get_text(" ", strip=True))
    return "\n".join(texts)

def _read_html(fp: Path) -> str:
    soup = BeautifulSoup(fp.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    return soup.get_text(" ", strip=True)

def load_text(file_path: Path) -> str:
    suf = file_path.suffix.lower()
    if suf in [".txt", ""]:
        return _read_txt(file_path)
    if suf == ".pdf":
        return _read_pdf(file_path)
    if suf == ".docx":
        return _read_docx(file_path)
    if suf == ".epub":
        return _read_epub(file_path)
    if suf in [".html", ".htm"]:
        return _read_html(file_path)
    raise HTTPException(400, f"不支持的格式: {suf}")

def chunk_text(s: str, size=1000, overlap=100) -> List[str]:
    s = re.sub(r"\s+", " ", s)
    out, i, n = [], 0, len(s)
    while i < n:
        out.append(s[i : i + size])
        i += max(1, size - overlap)
    return out

def simple_stats(text: str, topk=50) -> Dict:
    tags = jieba.analyse.extract_tags(text, topK=topk, withWeight=True)
    seg = [w for w in jieba.lcut(text) if 1 < len(w) <= 3 and re.match(r"[\u4e00-\u9fa5]+", w)]
    freq = {}
    for w in seg:
        freq[w] = freq.get(w, 0) + 1
    persons = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:topk]
    item_hits = {}
    for w in seg:
        if re.search(r"(术|诀|经|法|阵|掌|剑|丹|器|符|体|功|篇|卷)$", w):
            item_hits[w] = item_hits.get(w, 0) + 1
    items = sorted(item_hits.items(), key=lambda x: x[1], reverse=True)[:topk]
    return {
        "keywords": [{"word": w, "weight": float(sw)} for w, sw in tags],
        "persons": [{"name": w, "count": int(c)} for w, c in persons],
        "items_skills": [{"name": w, "count": int(c)} for w, c in items],
        "chars": len(text),
        "words_est": len(seg),
    }

def save_doc(text: str, filename_hint: Optional[str]) -> str:
    doc_id = uuid.uuid4().hex[:12]
    doc_dir = DATA_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "text.txt").write_text(text, encoding="utf-8")
    meta = {"doc_id": doc_id, "filename_hint": filename_hint, "length": len(text)}
    (doc_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    chunks = chunk_text(text, size=1200, overlap=100)
    (doc_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps({"i": i, "text": c}, ensure_ascii=False) for i, c in enumerate(chunks)),
        encoding="utf-8"
    )
    return doc_id

def read_doc(doc_id: str) -> Dict:
    doc_dir = DATA_DIR / doc_id
    if not doc_dir.exists():
        raise HTTPException(404, "doc_id 不存在")
    meta = json.loads((doc_dir / "meta.json").read_text(encoding="utf-8"))
    text = (doc_dir / "text.txt").read_text(encoding="utf-8")
    return {"meta": meta, "text": text}

@app.get("/openapi.json", include_in_schema=False)
async def overridden_openapi(request: Request):
    s = get_openapi(title=app.title, version=app.version, routes=app.routes,
                    description="Local corpus service for ChatGPT")
    base = str(request.base_url).rstrip("/")
    s["servers"] = [{"url": base}]
    return JSONResponse(s)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/docs_list")
def list_docs():
    out = []
    for p in DATA_DIR.glob("*/meta.json"):
        meta = json.loads(p.read_text(encoding="utf-8"))
        out.append(meta)
    out.sort(key=lambda m: m.get("doc_id"))
    return {"docs": out}

@app.post("/upload")
def analyze_upload(
    file: UploadFile = File(...),
    filename_hint: Optional[str] = Form(None),
    cap: int = Form(100000),
    preview: int = Form(150)
):
    suffix = Path(file.filename or (filename_hint or "")).suffix or ".txt"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(file.file.read()); tf.flush(); tf.close()
    text = load_text(Path(tf.name))
    os.unlink(tf.name)
    text_cap = text[:cap]
    doc_id = save_doc(text_cap, filename_hint or file.filename)
    stats = simple_stats(text_cap, topk=50)
    preview_chunks = chunk_text(text_cap, size=1000, overlap=0)[:max(1, preview // 1000)]
    return {"doc_id": doc_id, "filename": filename_hint or file.filename,
            "preview": preview_chunks, "stats": stats}

@app.post("/analyze")
def analyze_doc(doc_id: str = Form(...), cap: int = Form(200000), preview: int = Form(150)):
    d = read_doc(doc_id)
    text = d["text"][:cap]
    stats = simple_stats(text, topk=50)
    preview_chunks = chunk_text(text, size=1000, overlap=0)[:max(1, preview // 1000)]
    return {"doc_id": doc_id, "stats": stats, "preview": preview_chunks}

@app.get("/search")
def search(doc_id: str = Query(...), q: str = Query(...), top_k: int = Query(10)):
    d = read_doc(doc_id)
    chunks = []
    for line in (DATA_DIR / doc_id / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        score = obj["text"].count(q)
        if score > 0:
            chunks.append({"i": obj["i"], "score": score, "text": obj["text"][:300]})
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return {"doc_id": doc_id, "query": q, "hits": chunks[:top_k]}
