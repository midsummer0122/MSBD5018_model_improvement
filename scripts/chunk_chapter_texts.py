#!/usr/bin/env python3
"""处理 中央法规/0章条法律类 —— 章/节/条/项 多级结构化法条文本的父子块切分。

规则：
1. 标题 = 文件名去掉末尾所有括号及其内容（如 …(20181229) → …；…(2026修订)(FBM-CLI…) → …）。
2. 第 1、2 行相同则去重；"此复"及其后删除（本类为法条，通常无此复，保留无害）。
3. 结构解析（关键字匹配 + 边界守卫，逐级）：
   - 一级标题：第X章（后跟章标题，独占一行，如"第一章　总则"），标题内部全角空格归一。
   - 二级标题：第X节（少数文件有，如"第一节　一般规定"），同样可带节标题。
   - 三级标题：第X条（数字可到百位，如"第一百零七条"），后无标题直接接正文。
   - 四级标题：（一）（二）…。
   - 各标题前的直接文本 → 该级"概览"块（第 0 小点）：导言、章绪论、节绪论、条概览。
   边界守卫排除"本法第X条""依照第四十六条"等跨条文引用。
4. 每个块递归切分（段落边界 → 句边界 → 硬切；150–500 字，后块加 30–80 字 overlap，短块合并）。
5. 输出 JSON（doc_id/title/source_path/parents/children）到 knowledge_data/json/。
   父块 = 每个章（导言亦单独一个父块），子块 title_path 保留 章/节/条/项 全层级。

doc_id = 类编号(0) + 四位序号 → 00001、00002、……（按文件名排序依次编号）。

用法：
    python scripts/chunk_chapter_texts.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "knowledge_data" / "中央法规" / "0章条法律类"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "knowledge_data" / "json"

MIN_CHILD_CHARS = 150
MAX_CHILD_CHARS = 500
OVERLAP_MIN_CHARS = 30
OVERLAP_MAX_CHARS = 80

# 文档类编号（0 = 章条法律类），doc_id = 类编号 + 四位序号
CLASS_NUMBER = "0"

SENTENCE_END_RE = re.compile(r"[。！？；]+[”’」』】）)]*")
PARAGRAPH_SEPS = ("\n\n", "\n")

BOUNDARY_CHARS = "。！？；：\n　 \t"
# 中文数字（含百/零，覆盖"第一百零七条"）
CN_NUM = "一二三四五六七八九十百零〇"
CN_NUM_SMALL = "一二三四五六七八九十"

# 一级：第X章；二级：第X节；三级：第X条；四级：（一）
CHAP_RE = re.compile(rf"第[{CN_NUM}]{{1,3}}章")
JIE_RE = re.compile(rf"第[{CN_NUM}]{{1,3}}节")
TIAO_RE = re.compile(rf"第[{CN_NUM}]{{1,5}}条")
XIANG_RE = re.compile(rf"[（(][{CN_NUM_SMALL}]{{1,3}}[）)]")


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


def _extract_heading_title(raw_body: str) -> str:
    """提取章/节标题：标题独占一行（≤40字），内部全角空格归一（"总　则"→"总则"）。"""
    m = re.match(r"[　\s]*([^\n]{1,40})\n", raw_body)
    if m:
        t = re.sub(r"[\s　]+", "", m.group(1))
        if t:
            return t
    return ""


def _strip_first_line(text: str) -> str:
    nl = text.find("\n")
    return text[nl + 1 :].strip() if nl >= 0 else ""


def parse_blocks(title: str, text: str) -> list[tuple[str, str, str]]:
    """解析为块列表 [(父块 title_path, 块 title_path, 块文本)]。

    父块 title_path = 章级（导言 → "{title} > 导言"；其余 → "{title} > 第X章 标题"）。
    块 title_path = 全层级：章绪论 → "章 > 概览"；节绪论 → "章 > 节 > 概览"；
                    条概览 → "章[ > 节] > 条 > 概览"；项 → "章[ > 节] > 条 > （一）"。
    """
    markers: list[tuple[int, int, int, str]] = []
    for m in CHAP_RE.finditer(text):
        if _is_boundary(text, m.start()):
            markers.append((m.start(), m.end(), 1, m.group(0)))
    for m in JIE_RE.finditer(text):
        if _is_boundary(text, m.start()):
            markers.append((m.start(), m.end(), 2, m.group(0)))
    for m in TIAO_RE.finditer(text):
        if _is_boundary(text, m.start()):
            markers.append((m.start(), m.end(), 3, m.group(0)))
    for m in XIANG_RE.finditer(text):
        if _is_boundary(text, m.start()):
            markers.append((m.start(), m.end(), 4, m.group(0)))
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

    cur_chap: str | None = None  # 章父块 title_path
    cur_jie: str | None = None   # 节 label
    cur_tiao_prefix: str | None = None  # 当前条完整前缀（含章/节）

    for i, (start, end, level, marker) in enumerate(markers):
        next_start = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        raw_body = text[end:next_start]

        if level == 1:  # 章
            t = _extract_heading_title(raw_body)
            label = f"{marker} {t}" if t else marker
            cur_chap = f"{title} > {label}"
            cur_jie = None
            cur_tiao_prefix = None
            block_tp = f"{cur_chap} > 概览"
            body = _strip_first_line(raw_body) if t else raw_body.strip()
        elif level == 2:  # 节
            t = _extract_heading_title(raw_body)
            label = f"{marker} {t}" if t else marker
            cur_jie = label
            cur_tiao_prefix = None
            block_tp = f"{cur_chap} > {label} > 概览"
            body = _strip_first_line(raw_body) if t else raw_body.strip()
        elif level == 3:  # 条
            cur_tiao_prefix = f"{cur_chap} > {marker}" if not cur_jie else f"{cur_chap} > {cur_jie} > {marker}"
            block_tp = f"{cur_tiao_prefix} > 概览"
            body = raw_body.strip()
        else:  # 项
            block_tp = f"{cur_tiao_prefix} > {marker}" if cur_tiao_prefix else f"{cur_chap} > {marker}"
            body = raw_body.strip()

        if body:
            blocks.append((cur_chap or f"{title} > 导言", block_tp, body))
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


def merge_title_paths(left: str, right: str) -> str:
    """合并两个 title_path：共同前缀 + 不同尾部用 '；' 连接（保留条号/项号）。"""
    if not left:
        return right
    if not right or left == right:
        return left
    left_parts = [p.strip() for p in left.split(" > ") if p.strip()]
    right_parts = [p.strip() for p in right.split(" > ") if p.strip()]
    common_len = 0
    for a, b in zip(left_parts, right_parts):
        if a != b:
            break
        common_len += 1
    common = left_parts[:common_len]
    left_tail = " > ".join(left_parts[common_len:])
    right_tail = " > ".join(right_parts[common_len:])
    if not left_tail:
        merged_tail = right_tail
    elif not right_tail:
        merged_tail = left_tail
    elif right_tail in left_tail.split("；"):
        merged_tail = left_tail
    else:
        merged_tail = f"{left_tail}；{right_tail}"
    return " > ".join([*common, merged_tail]) if common else merged_tail


def merge_short_chunks(chunks: list[tuple[str, str]], min_chars: int = MIN_CHILD_CHARS) -> list[tuple[str, str]]:
    if len(chunks) <= 1:
        return chunks
    merged: list[tuple[str, str]] = []
    cur_tp, cur_text = chunks[0]
    for tp, text in chunks[1:]:
        if norm_len(cur_text) < min_chars:
            cur_text = cur_text + "\n" + text
            cur_tp = merge_title_paths(cur_tp, tp)
        else:
            merged.append((cur_tp, cur_text))
            cur_tp, cur_text = tp, text
    if merged and norm_len(cur_text) < min_chars:
        ptp, ptext = merged[-1]
        merged[-1] = (merge_title_paths(ptp, cur_tp), ptext + "\n" + cur_text)
    else:
        merged.append((cur_tp, cur_text))
    return merged


def build_document(path: Path, doc_id: str) -> dict:
    raw = read_text(path)
    text = clean_text(raw)
    title = title_from_filename(path)

    blocks = parse_blocks(title, text)

    # 按父块（章）分组，保持顺序
    sections: list[dict] = []
    section_index: dict[str, int] = {}
    for parent_tp, block_tp, block_text in blocks:
        if parent_tp not in section_index:
            section_index[parent_tp] = len(sections)
            sections.append({"title_path": parent_tp, "blocks": []})
        sections[section_index[parent_tp]]["blocks"].append((block_tp, block_text))

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
    parser = argparse.ArgumentParser(description="切分 中央法规/0章条法律类 为父子块 JSON。")
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
            f"[OK] {path.name[:30]}... -> parents={len(data['parents'])}, children={len(data['children'])}"
        )
        ok += 1

    print(f"done: ok={ok}, fail={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
