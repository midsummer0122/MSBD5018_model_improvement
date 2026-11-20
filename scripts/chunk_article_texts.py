#!/usr/bin/env python3
"""处理 中央法规/2条目类Ⅰ —— 一级标题(第X条)+二级标题(（一）) 结构化文本的父子块切分。

规则：
1. 标题 = 文件名去掉末尾所有括号及其内容（如 …(1995修订)(FBM-CLI.2.11532) → …）。
2. 第 1、2 行相同则去重；"此复"及其后删除（本类多为条例/规定，通常无此复，保留该步骤无害）。
3. 结构解析（关键字匹配 + 边界守卫）：
   - 一级标题：第一条、第二条、…第十一条…（第 + 中文数字 1–3 位 + 条），须前接边界。
   - 二级标题：（一）（二）…，同上须前接边界。
   - 第一条前的导言 → 「导言」块（第 0 大点）。
   - 一级标题下、首个二级标题前的直接文本 → 「概览」块（第 0 小点）。
   - 二级标题下的文本 → 各自块。
   边界守卫排除误匹配：如"本办法第八条""本规定第六条"等跨条文引用。
   注意：第X条/（一）后无标题、直接接正文，故本类不做标题提取，label 直接取条号/项号。
4. 每个块递归切分（段落边界 → 句边界 → 硬切；150–500 字，后块加 30–80 字 overlap，短块合并）。
5. 输出 JSON（doc_id/title/source_path/parents/children）到 knowledge_data/json/。

doc_id = 类编号(2) + 四位序号 → 20001、20002、……（按文件名排序依次编号）。

用法：
    python scripts/chunk_article_texts.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "knowledge_data" / "中央法规" / "2条目类Ⅰ"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "knowledge_data" / "json"

MIN_CHILD_CHARS = 150
MAX_CHILD_CHARS = 500
MIN_PARENT_CHARS = 1000
OVERLAP_MIN_CHARS = 30
OVERLAP_MAX_CHARS = 80

# 文档类编号（2 = 条目类Ⅰ），doc_id = 类编号 + 四位序号
CLASS_NUMBER = "2"

SENTENCE_END_RE = re.compile(r"[。！？；]+[”’」』】）)]*")
PARAGRAPH_SEPS = ("\n\n", "\n")

# 标题边界字符：标题前必须是句末标点/换行/空白，否则视为"本办法第X条"等跨条文引用
BOUNDARY_CHARS = "。！？；：\n　 \t"
# 一级标题：第X条（第 + 中文数字 1–3 位 + 条，覆盖到四十八等）
L1_RE = re.compile(r"第[一二三四五六七八九十]{1,3}条")
# 二级标题：（一）（二）…（全角/半角括号）
L2_RE = re.compile(r"[（(][一二三四五六七八九十]{1,3}[）)]")


def read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")


def norm_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def title_from_filename(path: Path) -> str:
    stem = path.stem
    title = re.sub(r"(?:[（(][^（）()]*[）)]\s*)+$", "", stem).strip()
    return title or stem


def relative_source_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    if len(lines) >= 2 and lines[0] == lines[1]:
        del lines[1]
    text = "\n".join(lines)
    idx = text.find("此复")
    if idx >= 0:
        text = text[:idx]
    return text.strip()


def _is_boundary(text: str, pos: int) -> bool:
    return pos == 0 or text[pos - 1] in BOUNDARY_CHARS


def parse_blocks(title: str, text: str) -> list[tuple[str, str, str]]:
    """解析为块列表 [(父块 title_path, 块 title_path, 块文本)]。

    父块 title_path：导言 → "{title} > 导言"；一级 → "{title} > 第X条"。
    块 title_path：导言 → 同父；一级概览 → 父 + " > 概览"；二级 → 父 + " > （一）"。
    """
    markers: list[tuple[int, int, int, str]] = []
    for m in L1_RE.finditer(text):
        if _is_boundary(text, m.start()):
            markers.append((m.start(), m.end(), 1, m.group(0)))
    for m in L2_RE.finditer(text):
        if _is_boundary(text, m.start()):
            markers.append((m.start(), m.end(), 2, m.group(0)))
    markers.sort(key=lambda x: x[0])

    blocks: list[tuple[str, str, str]] = []
    if not markers:
        if text.strip():
            ptp = f"{title} > 导言"
            blocks.append((ptp, ptp, text.strip()))
        return blocks

    intro = text[: markers[0][0]].strip()
    if intro:
        ptp = f"{title} > 导言"
        blocks.append((ptp, ptp, intro))

    cur_parent_tp: str | None = None
    for i, (start, end, level, marker) in enumerate(markers):
        next_start = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        body = text[end:next_start].strip()
        if level == 1:
            cur_parent_tp = f"{title} > {marker}"
            block_tp = f"{cur_parent_tp} > 概览"
        else:
            block_tp = f"{cur_parent_tp} > {marker}" if cur_parent_tp else f"{title} > {marker}"
        if body:
            blocks.append((cur_parent_tp or f"{title} > {marker}", block_tp, body))
    return blocks


def _hard_cut(text: str, max_chars: int) -> list[str]:
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def split_recursive(text: str, max_chars: int = MAX_CHILD_CHARS) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if norm_len(text) <= max_chars:
        return [text]

    for sep in PARAGRAPH_SEPS:
        if sep not in text:
            continue
        pieces = [p for p in text.split(sep) if p.strip()]
        if len(pieces) <= 1:
            continue
        chunks: list[str] = []
        buf = ""
        for p in pieces:
            merged = (buf + sep + p) if buf else p
            if buf and norm_len(merged) > max_chars:
                chunks.extend(split_recursive(buf, max_chars))
                buf = p
            else:
                buf = merged
        if buf:
            chunks.extend(split_recursive(buf, max_chars))
        return chunks

    matches = list(SENTENCE_END_RE.finditer(text))
    best: int | None = None
    target = len(text) / 2
    for m in matches:
        pos = m.end()
        if norm_len(text[:pos]) < MIN_CHILD_CHARS:
            continue
        if norm_len(text[pos:]) < MIN_CHILD_CHARS:
            continue
        if best is None or abs(pos - target) < abs(best - target):
            best = pos
    if best is not None:
        return split_recursive(text[:best], max_chars) + split_recursive(text[best:], max_chars)

    return _hard_cut(text, max_chars)


def overlap_prefix(prev_text: str) -> str:
    compact = re.sub(r"\s+", "", prev_text)
    if not compact:
        return ""
    sentences = [s.strip() for s in re.findall(r"[^。！？；\n]+[。！？；]?", prev_text) if s.strip()]
    prefix = ""
    for s in reversed(sentences):
        cand = s + prefix
        if norm_len(cand) <= OVERLAP_MAX_CHARS:
            prefix = cand
        else:
            if not prefix:
                prefix = s[-OVERLAP_MAX_CHARS:]
            break
    if norm_len(prefix) >= OVERLAP_MIN_CHARS:
        return prefix.strip()
    return compact[-OVERLAP_MAX_CHARS:]


def add_overlap(chunks: list[str]) -> list[str]:
    if len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        prefix = overlap_prefix(prev)
        if prefix and not cur.startswith(prefix):
            result.append(prefix + "\n" + cur)
        else:
            result.append(cur)
    return result


def _common_title_path(tp1: str, tp2: str) -> str:
    p1 = tp1.split(" > ")
    p2 = tp2.split(" > ")
    common: list[str] = []
    for a, b in zip(p1, p2):
        if a == b:
            common.append(a)
        else:
            break
    return " > ".join(common) if common else tp1


def merge_short_chunks(chunks: list[tuple[str, str]], min_chars: int = MIN_CHILD_CHARS) -> list[tuple[str, str]]:
    if len(chunks) <= 1:
        return chunks
    merged: list[tuple[str, str]] = []
    cur_tp, cur_text = chunks[0]
    for tp, text in chunks[1:]:
        if norm_len(cur_text) < min_chars:
            cur_text = cur_text + "\n" + text
            cur_tp = _common_title_path(cur_tp, tp)
        else:
            merged.append((cur_tp, cur_text))
            cur_tp, cur_text = tp, text
    if merged and norm_len(cur_text) < min_chars:
        ptp, ptext = merged[-1]
        merged[-1] = (_common_title_path(ptp, cur_tp), ptext + "\n" + cur_text)
    else:
        merged.append((cur_tp, cur_text))
    return merged


def build_document(path: Path, doc_id: str) -> dict:
    raw = read_text(path)
    text = clean_text(raw)
    title = title_from_filename(path)

    blocks = parse_blocks(title, text)

    sections: list[dict] = []
    section_index: dict[str, int] = {}
    for parent_tp, block_tp, block_text in blocks:
        if parent_tp not in section_index:
            section_index[parent_tp] = len(sections)
            sections.append({"title_path": parent_tp, "blocks": []})
        sections[section_index[parent_tp]]["blocks"].append((block_tp, block_text))

    # 每条（导言亦为一条）一个父块，保留条号；父块之间不做合并
    parents: list[dict] = []
    children: list[dict] = []
    child_seq = 0
    for sec in sections:
        parent_id = f"{doc_id}_p{len(parents) + 1:03d}"

        sec_children: list[tuple[str, str]] = []
        for block_tp, block_text in sec["blocks"]:
            parts = split_recursive(block_text)
            parts = add_overlap(parts)
            for part in parts:
                sec_children.append((block_tp, part))
        sec_children = merge_short_chunks(sec_children)

        child_ids: list[str] = []
        for block_tp, ctext in sec_children:
            child_seq += 1
            cid = f"{doc_id}_c{child_seq:03d}"
            child_ids.append(cid)
            children.append({"chunk_id": cid, "parent_id": parent_id, "title_path": block_tp, "text": ctext})

        parents.append(
            {
                "parent_id": parent_id,
                "title_path": sec["title_path"],
                "text": "\n".join(t for _, t in sec["blocks"]),
                "child_ids": child_ids,
            }
        )

    return {
        "doc_id": doc_id,
        "source_path": relative_source_path(path),
        "title": title,
        "parents": parents,
        "children": children,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="切分 中央法规/2条目类Ⅰ 为父子块 JSON。")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(args.input_dir.glob("*.txt"))
    if not files:
        print(f"[WARN] 未找到 txt 文件: {args.input_dir}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for idx, path in enumerate(files, start=1):
        doc_id = f"{CLASS_NUMBER}{idx:04d}"
        try:
            data = build_document(path, doc_id)
        except Exception as exc:
            print(f"[ERROR] {path.name}: {exc}")
            fail += 1
            continue
        out = args.output_dir / f"{path.stem}.json"
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[OK] {path.name[:32]}... -> parents={len(data['parents'])}, children={len(data['children'])}"
        )
        ok += 1

    print(f"done: ok={ok}, fail={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
