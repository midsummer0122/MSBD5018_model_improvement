"""Extract case PDFs and reuse the project's recursive parent/child splitting."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pymupdf

from chunk_natural_paragraph_texts import add_overlap, merge_short_chunks, norm_len, split_recursive

ROOT_DIR = Path(__file__).resolve().parents[1]


def extract_text(path: Path) -> str:
    with pymupdf.open(path) as pdf:
        text = "\n".join(page.get_text() for page in pdf)
    # The supplied exports put their advertising/copyright footer after the judgment.
    text = re.split(r"(?:©\s*)?北大法宝[：:]", text, maxsplit=1)[0]
    lines = [line.rstrip() for line in text.splitlines()]
    lines = [line for line in lines if not re.search(r"【法宝引证码】|www\.pkulaw\.(?:cn|com)", line)]
    text = "\n".join(lines).strip()
    if norm_len(text) < 100 or text.count("\ufffd") > len(text) * 0.01:
        raise ValueError(f"Text extraction requires inspection/OCR: {path.name}")
    return text


def build_case(path: Path, doc_id: str, input_dir: Path) -> dict:
    title = path.stem
    text = extract_text(path)
    authority = path.relative_to(input_dir).parts[0]
    parent_texts = merge_short_chunks(split_recursive(text, max_chars=3000), min_chars=1000)
    parents, children = [], []
    for parent_number, parent_text in enumerate(parent_texts, start=1):
        pid = f"{doc_id}_p{parent_number:03d}"
        child_texts = merge_short_chunks(add_overlap(split_recursive(parent_text, max_chars=420)))
        child_ids = []
        for child_text in child_texts:
            cid = f"{doc_id}_c{len(children) + 1:03d}"
            child_ids.append(cid)
            children.append({"chunk_id": cid, "parent_id": pid, "title_path": title,
                             "text": child_text, "doc_type": "case", "source_name": title,
                             "authority_level": authority})
        parents.append({"parent_id": pid, "title_path": title, "text": parent_text, "child_ids": child_ids})
    return {"doc_id": doc_id, "source_path": path.relative_to(ROOT_DIR).as_posix(),
            "title": title, "metadata": {"doc_type": "case", "authority_level": authority},
            "parents": parents, "children": children}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT_DIR / "knowledge_data/劳动案例")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "knowledge_data/json_cases")
    args = parser.parse_args()
    paths = sorted((p for p in args.input_dir.rglob("*") if p.suffix.lower() == ".pdf"
                    and "__MACOSX" not in p.parts and not p.name.startswith("._")),
                   key=lambda p: (p.name, p.as_posix()))
    if not paths:
        raise FileNotFoundError("No case PDFs found")
    # Extract everything before writing, so an unreadable PDF cannot silently leave a partial corpus.
    documents = [build_case(p, f"5{n:04d}", args.input_dir) for n, p in enumerate(paths, start=1)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for document in documents:
        out = args.output_dir / f"{document['doc_id']}.json"
        content = json.dumps(document, ensure_ascii=False, indent=2)
        if out.exists() and out.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing case output differs; inspect before replacing: {out.name}")
        out.write_text(content, encoding="utf-8")
    print(json.dumps({"documents": len(documents), "parents": sum(len(d['parents']) for d in documents),
                      "children": sum(len(d['children']) for d in documents),
                      "characters": sum(len(p['text']) for d in documents for p in d['parents'])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
