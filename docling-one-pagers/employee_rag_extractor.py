#!/usr/bin/env python3
"""
Merged extractor: rich non-experience fields + airtight experience

- Keeps the *full* section parsing from Docling (summary, skills, education, languages, certifications, clients, etc.).
- Replaces ONLY the 'experience' field using robust rules:
    * Parse experience from MD (debug text) and RAW TXT
    * Choose best among: RAW vs MD vs Docling (from parse_markdown)
    * Handle jobs-before-heading, same-line achievements, embedded headers, broken tokens (Node. + js → Node.js)
    * Keep clients out of experience
- Normalizes Deloitte title to exactly one of: Consultant | Analyst | Manager
- Processes all *.[Pp][Pp][Tt][Xx], continues on errors
- Exports docling markdown to out/debug/<name>.md for troubleshooting

"""

import re, os, json, shutil, argparse, logging, subprocess, html
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from docling.document_converter import DocumentConverter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("employee_rag")

# ----------------- Common helpers -----------------
BULLET_PREFIX_RE = re.compile(r"""^([\-–—•●▪◦*]+|\d+\.\s+|\d+\)\s+|[A-Za-z]\.\s+|[A-Za-z]\)\s+)""", re.VERBOSE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
MONTHS = ("jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec")
ROLE_KEYWORDS = ["engineer","consultant","developer","manager","analyst","architect","lead","designer","scientist","specialist"]

SECTION_ALIASES = {
    "summary of professional experience": "summary",
    "professional summary": "summary",
    "summary": "summary",
    "key skills": "key_skills",
    "skills": "key_skills",
    "business skills": "business_skills",
    "technology skills": "technology_skills",
    "industry experience": "industry_experience",
    "relevant experience": "experience",
    "work experience": "experience",
    "experience": "experience",
    "selected clients": "clients",
    "clients": "clients",
    "education": "education",
    "languages": "languages",
    "certifications": "certifications",
}

def unescape(x: str) -> str:
    return html.unescape(x)

def normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()

def normalize_dash(s: str) -> str:
    s = s.replace("—", "–")
    s = re.sub(r"\s*-\s*", " – ", s)
    s = re.sub(r"\s*–\s*", " – ", s)
    return re.sub(r"\s{2,}", " ", s).strip()

def looks_like_duration(s: str) -> bool:
    if not s: return False
    t = s.lower()
    if "present" in t or "current" in t: return True
    if re.search(r"(19|20)\d{2}", t): return True
    if any(m in t for m in MONTHS): return True
    if re.search(r"\b(19|20)\d{2}\s*[–—-]\s*(19|20)\d{2}\b", t): return True
    return False

# Strip simple markdown decorations (bold/italics/code/headers) so header regex works
def strip_md_decor(line: str) -> str:
    l = line.strip()
    # strip leading md markers
    l = re.sub(r"^[*_`>#]+", "", l)
    # strip trailing markers
    l = re.sub(r"[*_`]+$", "", l)
    # unwrap **text** or __text__
    if l.startswith("**") and l.endswith("**") and len(l) > 4:
        l = l[2:-2].strip()
    if l.startswith("__") and l.endswith("__") and len(l) > 4:
        l = l[2:-2].strip()
    return l.strip()

# ----------------- Docling conversion -----------------
def has_soffice() -> bool:
    return shutil.which("soffice") is not None

def pptx_to_pdf(pptx: Path, pdf_dir: Path) -> Optional[Path]:
    try:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        if not has_soffice():
            log.warning("LibreOffice 'soffice' not found on PATH; skipping PDF conversion.")
            return None
        cmd = ["soffice","--headless","--convert-to","pdf","--outdir",str(pdf_dir), str(pptx)]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_pdf = pdf_dir / f"{pptx.stem}.pdf"
        return out_pdf if out_pdf.exists() else None
    except subprocess.CalledProcessError as e:
        log.warning("PDF conversion failed for %s: %s", pptx.name, e); return None

def extract_docling_markdown(path: Path) -> str:
    conv = DocumentConverter()
    res = conv.convert(str(path))
    md = res.document.export_to_markdown() or ""
    txt = res.document.export_to_text() or ""
    return md if len(md) >= len(txt) else txt

# ----------------- RAW TXT IO -----------------
def _try_read_raw_full(pptx_path: Path, raw_txt_dir: Optional[Path]) -> Optional[str]:
    candidates=[f"{pptx_path.stem}_raw_text.txt", f"{pptx_path.stem}_extracted.txt", f"{pptx_path.stem}.txt"]
    search_dirs=[]
    if raw_txt_dir and isinstance(raw_txt_dir, Path) and raw_txt_dir.exists():
        search_dirs.append(raw_txt_dir)
    search_dirs.append(pptx_path.parent)
    for d in search_dirs:
        for c in candidates:
            p=(d / c)
            if p.exists():
                try:
                    return p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass
    return None

def _try_read_raw_first_line(pptx_path: Path, raw_txt_dir: Optional[Path]) -> Optional[str]:
    raw = _try_read_raw_full(pptx_path, raw_txt_dir)
    if not raw: return None
    for line in raw.splitlines():
        if line.strip(): return line.strip()
    return None

# ----------------- Non-experience parse from Docling (kept from original) -----------------
def _looks_like_person_name(text: str) -> bool:
    text = text.strip().strip(":").strip()
    if not text or len(text) > 80: return False
    if any(ch.isdigit() for ch in text): return False
    tokens = [t for t in text.split() if t]
    if not (2 <= len(tokens) <= 4): return False
    if text.lower() in SECTION_ALIASES: return False
    for t in tokens:
        t_clean = t.strip(",")
        for p in t_clean.split("-"):
            if re.match(r"^[A-Z][a-z'’]+$", p): break
            if re.match(r"^[A-Z]+$", p) and len(p)>1: break
        else:
            return False
    return True

def split_lines_keep_wrapped(md: str) -> List[str]:
    md = normalize_ws(md)
    raw = md.split("\n")
    out: List[str] = []
    for line in raw:
        l = line.strip()
        if not l:
            out.append(""); continue
        if BULLET_PREFIX_RE.match(l):
            out.append(l); continue
        prev = out[-1] if out else ""
        prev_is_job = bool(prev and _is_job_header(prev))
        prev_is_section = prev.lower().strip(": ") in SECTION_ALIASES
        if out and out[-1] and not BULLET_PREFIX_RE.match(out[-1]) and not prev_is_job and not prev_is_section:
            sep = " " if not out[-1].endswith((" ", "-", "–", "—")) else ""
            out[-1] = f"{out[-1]}{sep}{l}"
        else:
            out.append(l)
    return out

def extract_list_block(lines: List[str], start_idx: int) -> Tuple[List[str], int]:
    items: List[str] = []
    i = start_idx
    while i < len(lines):
        l = lines[i].strip()
        if not l: i += 1; continue
        if not BULLET_PREFIX_RE.match(l): break
        item = BULLET_PREFIX_RE.sub("", l).strip(" -–—")
        if item: items.append(unescape(item))
        i += 1
    return items, i

def sentences_to_items(text: str) -> List[str]:
    text = re.sub(r"\s*\.\s*", ". ", text).strip()
    parts = [p.strip(" •-–—") for p in re.split(r"\.\s+", text) if p.strip()]
    items = [(p if p.endswith(".") else p + ".") for p in parts]
    return [unescape(i) for i in items]

def is_section_heading(line: str, current_section: Optional[str]) -> bool:
    low = strip_md_decor(line).lower().strip(": ")
    if low in SECTION_ALIASES: return True
    if current_section == "experience": return False
    if line.lstrip().startswith(("#","##","###")): return True
    # NOTE: removed "short line without punctuation" heuristic to avoid dropping achievements
    return False

def normalize_heading(line: str) -> str:
    h = strip_md_decor(line).lower().strip(": ")
    return SECTION_ALIASES.get(h, h)

def extract_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None

def extract_name_and_title(lines: List[str], raw_first_line: Optional[str] = None) -> Tuple[str,str]:
    name=""; title=""
    scan = [strip_md_decor(l.strip()) for l in lines[:25] if l.strip()]
    for idx,l in enumerate(scan):
        if _looks_like_person_name(l):
            name=l
            for j in range(idx + 1, min(idx + 8, len(scan))):
                cand = scan[j]
                if len(cand)<=60 and any(k in cand.lower() for k in ROLE_KEYWORDS):
                    title=cand; break
            break
    if not name and raw_first_line:
        for cand in [x.strip() for x in raw_first_line.splitlines()[:5] if x.strip()]:
            if _looks_like_person_name(cand): name=cand; break
    if not title:
        for l in scan[:12]:
            if len(l)<=60 and any(k in l.lower() for k in ROLE_KEYWORDS):
                title=l; break
    return unescape(name), unescape(title)

def _is_job_header(line: str) -> Optional[Dict[str, str]]:
    JOB_HEADER_PATTERNS = [
        re.compile(r"^(?P<role>[^,–—()]+?),\s*(?P<company>[^()]+?)\s*\((?P<duration>[^)]+)\)\.?\s*$"),
        re.compile(r"^(?P<role>[^,–—()]+?)[–—-]\s*(?P<company>[^()]+?)\s*\((?P<duration>[^)]+)\)\.?\s*$"),
        re.compile(r"^(?P<role>[^()]+?)\s+at\s+(?P<company>[^()]+?)\s*\((?P<duration>[^)]+)\)\.?\s*$", re.IGNORECASE),
    ]
    s = strip_md_decor(line)
    for pat in JOB_HEADER_PATTERNS:
        m = pat.match(s)
        if m:
            role = m.group("role").strip(" -–—")
            company = m.group("company").strip(" -–—")
            duration = m.group("duration").strip()
            if looks_like_duration(duration) and re.search(r"[A-Za-z]", role):
                return {"role": role, "company": company, "duration": duration}
    return None

def consume_until_next_heading(lines: List[str], i: int, current_section: Optional[str]) -> Tuple[str,int]:
    buf: List[str] = []
    while i < len(lines):
        l = lines[i].strip()
        if not l: i+=1; continue
        if is_section_heading(l, current_section): break
        if _is_job_header(l): break
        if BULLET_PREFIX_RE.match(l): break
        buf.append(strip_md_decor(l)); i+=1
    return (unescape(" ".join(buf).strip()), i)

def parse_markdown(md: str, raw_first_line: Optional[str]=None) -> Dict[str,Any]:
    """Full record parser from Docling (kept intact); experience will be overridden later."""
    lines = split_lines_keep_wrapped(md)
    email = extract_email("\n".join(lines))
    name, title = extract_name_and_title(lines, raw_first_line=raw_first_line)

    out: Dict[str,Any] = {
        "name": name or None,
        "title": title or None,
        "contact": {"email": email} if email else {},
        "summary": "",
        "key_skills": {"business": [], "technology": [], "industry": []},
        "education": [], "languages": [], "certifications": [],
        "experience": [], "clients": []
    }

    i=0; current_section=None
    while i < len(lines):
        line = lines[i].strip()
        if not line: i+=1; continue

        job_global = _is_job_header(line)
        if job_global:
            current_section = "experience"
            out["experience"].append({**job_global, 'achievements': []})
            i += 1
            para, i2 = consume_until_next_heading(lines, i, current_section)
            if para:
                out["experience"][-1]["achievements"].extend(sentences_to_items(para))
                i = i2
            if i < len(lines) and BULLET_PREFIX_RE.match(lines[i].strip()):
                items, i = extract_list_block(lines, i)
                out["experience"][-1]["achievements"].extend(items)
            continue

        if is_section_heading(line, current_section):
            current_section = normalize_heading(line); i += 1; continue

        if current_section=="summary":
            text,i2 = consume_until_next_heading(lines,i,current_section)
            if i2==i: i+=1; continue
            if text: out["summary"]=text
            i=i2; continue

        if current_section in ("key_skills","business_skills","technology_skills","industry_experience"):
            low = strip_md_decor(line).lower().strip(": ")
            if low in ("business skills","technology skills","industry experience"):
                current_section = normalize_heading(low); i+=1; continue
            if BULLET_PREFIX_RE.match(line):
                items,i = extract_list_block(lines,i)
                target={"business_skills":"business","technology_skills":"technology","industry_experience":"industry"}.get(current_section,"industry")
                out["key_skills"].setdefault(target,[]).extend(items); continue
            i+=1; continue

        if current_section in ("education","languages","certifications","clients"):
            if BULLET_PREFIX_RE.match(line):
                items,i = extract_list_block(lines,i)
                out[current_section].extend(items); continue
            text,i2 = consume_until_next_heading(lines,i,current_section)
            if i2==i: i+=1; continue
            if text:
                if current_section=="clients":
                    out["clients"].extend([unescape(c.strip()) for c in strip_md_decor(text).split(",") if c.strip()])
                elif current_section=="languages":
                    out["languages"].extend([unescape(w.strip()) for w in strip_md_decor(text).split(",") if w.strip()])
                else:
                    out[current_section].append(unescape(strip_md_decor(text)))
            i=i2; continue

        if current_section=="experience":
            if BULLET_PREFIX_RE.match(line):
                items,i = extract_list_block(lines,i)
                if not out["experience"]:
                    out["experience"].append({"role":None,"company":None,"duration":None,"achievements":[]})
                out["experience"][-1]["achievements"].extend(items); continue

            text,i2 = consume_until_next_heading(lines,i,current_section)
            if i2==i: i+=1; continue
            if text:
                if not out["experience"]:
                    out["experience"].append({"role":None,"company":None,"duration":None,"achievements":[]})
                out["experience"][-1]["achievements"].extend(sentences_to_items(text))
            i=i2; continue

        i+=1

    # cleanup
    out["key_skills"] = {k:v for k,v in out["key_skills"].items() if v}
    for k in ("education","languages","certifications","experience","clients"):
        if not out[k]: del out[k]
    if not out.get("contact"): out.pop("contact", None)
    if not out.get("summary"): out.pop("summary", None)
    if not out.get("key_skills"): out.pop("key_skills", None)

    return out

# ----------------- Robust experience parsers (from MD/RAW) -----------------
LOOSE_HEADER_PATTERNS = [
    re.compile(r"^(?P<role>[^,–—()]+?),\s*(?P<company>[^()]+?)\s*\((?P<duration>[^)]+)\)\.?\s*(?P<after>.*)$"),
    re.compile(r"^(?P<role>[^,–—()]+?)[–—-]\s*(?P<company>[^()]+?)\s*\((?P<duration>[^)]+)\)\.?\s*(?P<after>.*)$"),
    re.compile(r"^(?P<role>[^()]+?)\s+at\s+(?P<company>[^()]+?)\s*\((?P<duration>[^)]+)\)\.?\s*(?P<after>.*)$", re.IGNORECASE),
]

def detect_job_with_trailing(line: str) -> Tuple[Optional[Dict[str,str]], Optional[str]]:
    s = strip_md_decor(line)
    for pat in LOOSE_HEADER_PATTERNS:
        m = pat.match(s)
        if m:
            role = m.group("role").strip(" -–—")
            company = m.group("company").strip(" -–—")
            duration = m.group("duration").strip()
            after = (m.group("after") or "").strip()
            if looks_like_duration(duration) and re.search(r"[A-Za-z]", role):
                return {"role": role, "company": company, "duration": duration}, (after if after else None)
    return None, None

def parse_experience_from_md(md_text: str) -> List[Dict[str, Any]]:
    md_text = normalize_ws(md_text)
    lines = [ln.strip() for ln in md_text.splitlines()]
    out = []; i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line: i += 1; continue
        hdr, trailing = detect_job_with_trailing(line)
        if not hdr:
            i += 1; continue
        job = {"role": hdr["role"], "company": hdr["company"], "duration": normalize_dash(unescape(hdr["duration"])), "achievements": []}
        if trailing:
            for ch in re.split(r"\s*;\s*|\s{2,}", trailing):
                ch = ch.strip(" •-–—")
                if ch: job["achievements"].append(ch if ch.endswith(".") else ch + ".")
        i += 1
        while i < len(lines):
            t = lines[i].strip()
            if not t: i += 1; continue
            t_nb = BULLET_PREFIX_RE.sub("", t).strip()
            # stop only at next job header or a real (non-experience) section heading
            if detect_job_with_trailing(t)[0] or detect_job_with_trailing(t_nb)[0]: break
            low = strip_md_decor(t).lower().strip(": ")
            if low in SECTION_ALIASES and SECTION_ALIASES[low] != "experience": break
            if BULLET_PREFIX_RE.match(t):
                item = BULLET_PREFIX_RE.sub("", t).strip(" -–—")
                if item: job["achievements"].append(item if item.endswith(".") else item + ".")
            else:
                parts = [x.strip() for x in re.split(r"\.\s+", strip_md_decor(t)) if x.strip()]
                for x in parts:
                    job["achievements"].append(x if x.endswith(".") else x + ".")
            i += 1
        # repair Node./React./Next. + "js ..."
        rep=[]; j=0
        while j < len(job["achievements"]):
            s = job["achievements"][j].rstrip("."); merged=False
            if j+1 < len(job["achievements"]):
                nx = job["achievements"][j+1].lstrip()
                if s.endswith(("Node","Node.","React","React.","Next","Next.")) and nx.lower().startswith("js"):
                    s = s.rstrip(".") + ".js" + (" " + nx[2:].lstrip() if len(nx)>2 else "")
                    j += 1; merged=True
            rep.append(s if s.endswith(".") else s + ".")
            if not merged: j += 1
        seen=set(); dedup=[]
        for a in rep:
            if a not in seen: seen.add(a); dedup.append(a)
        job["achievements"] = dedup
        if looks_like_duration(job["duration"]):
            out.append(job)

    # split any achievement that embeds another header
    repaired: List[Dict[str,Any]] = []
    for job in out:
        acc=[]; pending=None
        for ach in job["achievements"]:
            hdr, tr = detect_job_with_trailing(ach)
            if hdr:
                if acc:
                    jcopy = dict(job); jcopy["achievements"] = acc; repaired.append(jcopy); acc=[]
                newj = {"role": hdr["role"], "company": hdr["company"], "duration": hdr["duration"], "achievements": []}
                if tr: newj["achievements"].append(tr if tr.endswith(".") else tr + ".")
                pending = newj
            else:
                if pending: pending["achievements"].append(ach)
                else: acc.append(ach)
        if pending:
            if acc:
                jcopy = dict(job); jcopy["achievements"] = acc; repaired.append(jcopy)
            repaired.append(pending)
        else:
            repaired.append(job)
    return repaired

def find_sections(raw_text: str) -> Dict[str, Tuple[int,int]]:
    lines = [ln.rstrip() for ln in raw_text.splitlines()]
    indices: List[Tuple[int,str]] = []
    for i, ln in enumerate(lines):
        low = strip_md_decor(ln).lower().strip(": ").strip()
        if low in SECTION_ALIASES:
            indices.append((i, SECTION_ALIASES[low]))
    sections: Dict[str, Tuple[int,int]] = {}
    for idx, (start, name) in enumerate(indices):
        end = len(lines)
        if idx + 1 < len(indices):
            end = indices[idx+1][0]
        if name in sections:
            s0, e0 = sections[name]
            sections[name] = (min(s0, start), max(e0, end))
        else:
            sections[name] = (start+1, end)
    return sections

def parse_experience_from_raw(raw_text: Optional[str]) -> List[Dict[str, Any]]:
    if not raw_text: return []
    raw_text = normalize_ws(raw_text)
    sections = find_sections(raw_text)
    lines_all = [ln.strip() for ln in raw_text.splitlines()]
    start_end = sections.get("experience", None)
    if start_end:
        start, end = start_end
        before = "\n".join(lines_all[:start])
        lines = lines_all if re.search(r"\([^)]+\)", before) else lines_all[start:end]
    else:
        lines = lines_all
    return parse_experience_from_md("\n".join(lines))

# ----------------- Deloitte title normalization -----------------
def infer_deloitte_title(record: Dict[str,Any], raw_full: Optional[str]) -> Optional[str]:
    allowed = {"manager":"Manager", "consultant":"Consultant", "analyst":"Analyst"}
    t = (record.get("title") or "").lower()
    for k,v in allowed.items():
        if k in t: return v
    for job in record.get("experience", []):
        r = (job.get("role") or "").lower()
        for k,v in allowed.items():
            if k in r: return v
    if raw_full:
        head = "\n".join(raw_full.splitlines()[:40]).lower()
        for k,v in allowed.items():
            if f" {k}" in head or head.startswith(k):
                return v
    return None

# ----------------- Choose best experience -----------------
def score_exp(exp: List[Dict[str,Any]]) -> Tuple[int,int]:
    return (len(exp), sum(len(j.get("achievements", [])) for j in exp))

def choose_experience_best(raw_exp, md_exp, doc_exp) -> Tuple[List[Dict[str,Any]], str]:
    # Prefer highest (jobs, achievements). Tie-breaker: RAW > MD > Docling
    candidates = [("RAW", raw_exp), ("MD", md_exp), ("Docling", doc_exp)]
    best = max(candidates, key=lambda kv: (len(kv[1]), sum(len(j.get("achievements", [])) for j in kv[1]), {"RAW":3,"MD":2,"Docling":1}[kv[0]]))
    return best[1], best[0]

# ----------------- Pipeline -----------------
def process_one(pptx_path: Path, out_dir: Path, raw_txt_dir: Optional[Path]) -> Optional[Dict[str,Any]]:
    debug_dir = out_dir / "debug"; debug_dir.mkdir(exist_ok=True)
    pdf_dir = out_dir / "pdf"; pdf_dir.mkdir(exist_ok=True)

    pdf_path = pptx_to_pdf(pptx_path, pdf_dir)
    parse_source = pdf_path if pdf_path and pdf_path.exists() else pptx_path
    md = extract_docling_markdown(parse_source)
    if not md.strip():
        log.error("Docling extracted empty text for %s", pptx_path.name); return None
    (debug_dir / f"{pptx_path.stem}.md").write_text(md, encoding="utf-8")

    raw_full = _try_read_raw_full(pptx_path, raw_txt_dir)
    raw_first_line = _try_read_raw_first_line(pptx_path, raw_txt_dir)

    # 1) Parse full record (non-experience) from Docling
    record = parse_markdown(md, raw_first_line=raw_first_line)
    doc_exp = list(record.get("experience", []))

    # 2) Parse robust experiences
    md_exp  = parse_experience_from_md(md)
    raw_exp = parse_experience_from_raw(raw_full)

    # 3) Choose best and override 'experience' only
    chosen_exp, src = choose_experience_best(raw_exp, md_exp, doc_exp)
    if chosen_exp:
        record["experience"] = chosen_exp

    # 4) Normalize Deloitte title
    dt = infer_deloitte_title(record, raw_full)
    if dt: record["deloitte_title"] = dt

    record["_debug_experience_source"] = f"(experience from {src})"

    out_json = out_dir / f"{pptx_path.stem}.json"
    out_json.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("✅ %s -> %s %s", pptx_path.name, out_json.name, record["_debug_experience_source"])

    # Clean PDF if created
    if pdf_path:
        try: pdf_path.unlink(missing_ok=True)
        except Exception: pass
    return record

def process_folder(pptx_dir: Path, out_dir: Path, raw_txt_dir: Optional[Path]) -> List[Dict[str,Any]]:
    pptx_files = sorted(list(pptx_dir.glob("*.[Pp][Pp][Tt][Xx]")))
    if not pptx_files:
        log.warning("No PPTX files found in %s", pptx_dir); return []
    all_records=[]
    for p in pptx_files:
        try:
            rec = process_one(p, out_dir, raw_txt_dir)
            if rec: all_records.append(rec)
        except Exception as e:
            log.error("❌ Failed to process %s: %s", p.name, e)
            continue
    if all_records:
        (out_dir / "all_employees.json").write_text(json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Saved combined: all_employees.json")
    return all_records

def main():
    ap = argparse.ArgumentParser(description="Merged employee JSON extractor (rich non-experience + robust experience).")
    ap.add_argument("input_dir", help="Folder containing .pptx files")
    ap.add_argument("--out", default="json_output", help="Output folder (default: json_output)")
    ap.add_argument("--raw_txt_dir", default=None, help="Optional folder with raw TXT files")
    args = ap.parse_args()
    in_dir = Path(args.input_dir); out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)
    raw_txt_dir = Path(args.raw_txt_dir) if args.raw_txt_dir else None
    if raw_txt_dir and not raw_txt_dir.exists():
        log.warning("--raw_txt_dir %s does not exist; continuing without TXT fallback.", raw_txt_dir)
        raw_txt_dir = None
    process_folder(in_dir, out_dir, raw_txt_dir)

if __name__ == "__main__":
    main()
