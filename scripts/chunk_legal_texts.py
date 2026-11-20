#!/usr/bin/env python3
"""法律文档切分脚本。

把 ``knowledge_data/法条/`` 和 ``knowledge_data/案例/`` 下的原始 txt 切分成父子块，
输出 JSON 到 ``knowledge_data/json/``，结构与现有检索/入库脚本约定保持一致：

.. code-block:: json

    {
      "doc_id": "001",
      "source_path": "knowledge_data/法条/001_劳动合同法.txt",
      "title": "劳动合同法",
      "parents": [{"parent_id": "001_p001", "title_path": "...", "text": "...", "child_ids": ["001_c001"]}],
      "children": [{"chunk_id": "001_c001", "parent_id": "001_p001", "title_path": "...", "text": "..."}]
    }

切分规则：
- 法条：按「编 → 章 → 节 → 条」切分，以「条」为子块单元，章/节为父块；title_path 形如 ``劳动合同法 > 第三章 > 第四十六条``。
- 案例：按「当事人 / 基本案情 / 争议焦点 / 裁判理由 / 裁判结果 / 裁判依据」等切分。
- 子块约 150–500 字、父块约 1000–3000 字（软约束，超长按句边界硬切，过短做合并）。

用法：
    python scripts/chunk_legal_texts.py                  # 处理法条与案例两个目录
    python scripts/chunk_legal_texts.py --mode statute   # 仅处理法条
    python scripts/chunk_legal_texts.py --mode case      # 仅处理案例
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT_DIR / "knowledge_data"
DEFAULT_STATUTE_DIR = KNOWLEDGE_DIR / "法条"
DEFAULT_CASE_DIR = KNOWLEDGE_DIR / "案例"
DEFAULT_JSON_DIR = KNOWLEDGE_DIR / "json"

# 父子块长度软约束
CHILD_MIN = 150
CHILD_MAX = 500
PARENT_MAX = 3000

# 法条结构：第X编/章/节 作为父块边界，第X条 作为子块单元
HEADING_RE = re.compile(r"^\s*第[零一二三四五六七八九十百千万0-9]+[编章节]")
ARTICLE_RE = re.compile(r"^\s*第[零一二三四五六七八九十百千万0-9]+条")

# 案例结构：常见段落标题
CASE_SECTION_RE = re.compile(
    r"^\s*[（(【\[]?[一二三四五六七八九十百0-9]*[、.．\s）)\]\】]*"
    r"(当事人|基本案情|案件事实|争议焦点|裁判理由|本院认为|裁判结果|裁判依据|裁判要旨|审理经过|原告诉称|被告辩称|相关法条)"
)


def split_long_text(text: str, max_len: int) -> list[str]:
    """按句边界把超长文本切成 <= max_len 的片段；单句超长时硬切。"""
    text = re.sub(r"\s+", "\n", text).strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    sentences = re.split(r"(?<=[。！？；])\s*", text)
    chunks: list[str] = []
    buf = ""
    for raw in sentences:
        seg = raw.strip()
        if not seg:
            continue
        if len(seg) > max_len:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(seg), max_len):
                chunks.append(seg[i : i + max_len])
        elif len(buf) + len(seg) <= max_len:
            buf += seg
        else:
            if buf:
                chunks.append(buf)
            buf = seg
    if buf:
        chunks.append(buf)
    return chunks


def merge_short_text_chunks(chunks: list[str], min_len: int) -> list[str]:
    """把相邻的过短片段合并到 >= min_len（无法满足时保持原样）。"""
    if not chunks:
        return chunks
    merged: list[str] = []
    buf = ""
    for chunk in chunks:
        if not buf:
            buf = chunk
        elif len(buf) < min_len:
            buf += "\n" + chunk
        else:
            merged.append(buf)
            buf = chunk
    if buf:
        merged.append(buf)
    return merged


def doc_id_from_filename(stem: str) -> str:
    m = re.match(r"^(\d+)", stem)
    return m.group(1) if m else stem


def title_from_filename(stem: str) -> str:
    m = re.match(r"^\d+[_\-\s]*(.*)$", stem)
    return m.group(1) if m and m.group(1) else stem


def parse_statute(text: str, title: str) -> list[dict]:
    """把法条文本解析为 section 列表：{"path": ..., "articles": [(条号, 正文), ...]}"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sections: list[dict] = []
    current_path = title
    current_articles: list[tuple[str, str]] = []
    current_label: str | None = None
    current_lines: list[str] = []

    def flush_article() -> None:
        nonlocal current_label, current_lines
        if current_label is not None:
            current_articles.append((current_label, "\n".join(current_lines)))
            current_label = None
            current_lines = []

    def flush_section() -> None:
        nonlocal current_articles
        if current_articles:
            sections.append({"path": current_path, "articles": current_articles})
            current_articles = []

    for line in lines:
        if HEADING_RE.match(line):
            flush_article()
            flush_section()
            current_path = f"{title} > {line}" if line != title else title
        elif ARTICLE_RE.match(line):
            flush_article()
            current_label = ARTICLE_RE.match(line).group(0)
            current_lines = [line]
        else:
            if current_label is not None:
                current_lines.append(line)
            # 首条之前的导语等直接忽略，不作为子块

    flush_article()
    flush_section()
    return sections


def parse_case(text: str) -> list[tuple[str, str]]:
    """把案例文本按段落标题切分为 (段落名, 段落正文) 列表。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sections: list[tuple[str, str]] = []
    current_name = "案件全文"
    current_lines: list[str] = []
    for line in lines:
        if CASE_SECTION_RE.match(line):
            if current_lines:
                sections.append((current_name, "\n".join(current_lines)))
            current_name = CASE_SECTION_RE.match(line).group(1)
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_name, "\n".join(current_lines)))
    return sections


def _bucketize(child_specs: list[dict], max_len: int) -> list[list[dict]]:
    """把子块按顺序分组为若干父块，每个父块累计长度 <= max_len。"""
    buckets: list[list[dict]] = []
    current: list[dict] = []
    current_len = 0
    for spec in child_specs:
        if current and current_len + len(spec["text"]) > max_len:
            buckets.append(current)
            current = []
            current_len = 0
        current.append(spec)
        current_len += len(spec["text"])
    if current:
        buckets.append(current)
    return buckets


def build_document(doc_id: str, title: str, parent_specs: list[dict]) -> tuple[list[dict], list[dict]]:
    """根据 parent_specs 生成带 ID 的 parents/children。

    parent_specs 元素结构：{"title_path": str, "children": [{"title_path": str, "text": str}, ...]}
    """
    parents: list[dict] = []
    children: list[dict] = []
    child_seq = 0
    for p_idx, pspec in enumerate(parent_specs, start=1):
        parent_id = f"{doc_id}_p{p_idx:03d}"
        child_ids: list[str] = []
        parent_texts: list[str] = []
        for cspec in pspec["children"]:
            child_seq += 1
            child_id = f"{doc_id}_c{child_seq:03d}"
            child_ids.append(child_id)
            parent_texts.append(cspec["text"])
            children.append(
                {
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "title_path": cspec["title_path"],
                    "text": cspec["text"],
                }
            )
        parents.append(
            {
                "parent_id": parent_id,
                "title_path": pspec["title_path"],
                "text": "\n".join(parent_texts),
                "child_ids": child_ids,
            }
        )
    return parents, children


def build_statute_doc(doc_id: str, title: str, text: str) -> tuple[list[dict], list[dict]]:
    sections = parse_statute(text, title)
    if not sections:
        sections = [{"path": title, "articles": [(title, text)]}]

    parent_specs: list[dict] = []
    for section in sections:
        section_path = section["path"]

        # 以「条」为子块单元，超长按句边界切，过短相邻合并
        article_units: list[tuple[str, str]] = []
        for label, body in section["articles"]:
            for piece in split_long_text(body, CHILD_MAX):
                article_units.append((label, piece))

        child_specs: list[dict] = []
        i = 0
        while i < len(article_units):
            label, piece = article_units[i]
            if len(piece) >= CHILD_MIN:
                child_specs.append({"title_path": f"{section_path} > {label}", "text": piece})
                i += 1
                continue

            labels = [label]
            texts = [piece]
            total = len(piece)
            i += 1
            while i < len(article_units) and total < CHILD_MIN:
                if total + len(article_units[i][1]) > CHILD_MAX:
                    break
                labels.append(article_units[i][0])
                texts.append(article_units[i][1])
                total += len(texts[-1])
                i += 1
            child_tp = f"{section_path} > {labels[0]}" if len(labels) == 1 else section_path
            child_specs.append({"title_path": child_tp, "text": "\n".join(texts)})

        for bucket in _bucketize(child_specs, PARENT_MAX):
            parent_specs.append({"title_path": section_path, "children": bucket})

    return build_document(doc_id, title, parent_specs)


def build_case_doc(doc_id: str, title: str, text: str) -> tuple[list[dict], list[dict]]:
    sections = parse_case(text)
    if not sections:
        sections = [("案件全文", text)]

    child_specs: list[dict] = []
    for name, body in sections:
        pieces = split_long_text(body, CHILD_MAX)
        pieces = merge_short_text_chunks(pieces, CHILD_MIN)
        for piece in pieces:
            child_specs.append({"title_path": f"{title} > {name}", "text": piece})

    parent_specs = [{"title_path": title, "children": bucket} for bucket in _bucketize(child_specs, PARENT_MAX)]
    return build_document(doc_id, title, parent_specs)


def main() -> int:
    parser = argparse.ArgumentParser(description="把法律原始 txt 切分为父子块 JSON。")
    parser.add_argument("--mode", choices=("auto", "statute", "case"), default="auto")
    parser.add_argument("--statute-dir", type=Path, default=DEFAULT_STATUTE_DIR)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_JSON_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files: list[tuple[Path, str]] = []
    if args.mode in ("auto", "statute"):
        files += [(p, "statute") for p in sorted(args.statute_dir.glob("*.txt"))]
    if args.mode in ("auto", "case"):
        files += [(p, "case") for p in sorted(args.case_dir.glob("*.txt"))]

    if not files:
        print(
            f"[WARN] 未找到待切分的 txt 文件。请把原始文本放到 "
            f"{DEFAULT_STATUTE_DIR} 或 {DEFAULT_CASE_DIR} 下。"
        )
        return 0

    for path, kind in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig")

        doc_id = doc_id_from_filename(path.stem)
        title = title_from_filename(path.stem)
        source_path = str(path.relative_to(ROOT_DIR)).replace("\\", "/")

        if kind == "statute":
            parents, children = build_statute_doc(doc_id, title, text)
        else:
            parents, children = build_case_doc(doc_id, title, text)

        doc = {
            "doc_id": doc_id,
            "source_path": source_path,
            "title": title,
            "parents": parents,
            "children": children,
        }
        out = args.output_dir / f"{path.stem}.json"
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {path.name} -> {out.name} (parents={len(parents)}, children={len(children)})")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
