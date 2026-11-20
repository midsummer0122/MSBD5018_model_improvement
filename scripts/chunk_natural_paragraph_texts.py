#!/usr/bin/env python3
"""处理 中央法规/1自然段文本类 —— 自由文本（批复/答复/复函/通知等）的父子块切分。

规则（与用户约定一致）：
1. 标题 = 文件名去掉末尾括号及其内容（FBM-CLI 编号）。
2. 第 1、2 行相同则去重（保留一份）。
3. "此复"及其后内容全部删除。
4. 递归切分：段落边界（空行 / 换行）→ 句边界（。！？；）→ 硬切；
   子块 150–500 字，后块加 30–80 字前文 overlap，短块向前合并。
5. 输出 JSON（doc_id / title / source_path / parents / children）到 knowledge_data/json/，
   格式与 knowledge_data/文档切分参考 一致。

doc_id 约定：类编号 + 四位序号。1自然段文本类 → 10001、10002、……（按文件名排序依次编号）。

用法：
    python scripts/chunk_natural_paragraph_texts.py
    python scripts/chunk_natural_paragraph_texts.py --input-dir ... --output-dir ...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "knowledge_data" / "中央法规" / "1自然段文本类"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "knowledge_data" / "json"

MIN_CHILD_CHARS = 150
MAX_CHILD_CHARS = 500
OVERLAP_MIN_CHARS = 30
OVERLAP_MAX_CHARS = 80

# 文档类编号（1 = 自然段文本类），doc_id = 类编号 + 四位序号
CLASS_NUMBER = "1"

# 句末标点（含可选的右引号 / 右括号）
SENTENCE_END_RE = re.compile(r"[。！？；]+[”’」』】）)]*")
# 段落分隔符：空行优先，其次换行（本类文档每行即一个自然段，无硬折行）
PARAGRAPH_SEPS = ("\n\n", "\n")


def read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")


def norm_len(text: str) -> int:
    """去空白后的字符数（中文语义长度）。"""
    return len(re.sub(r"\s+", "", text))


def title_from_filename(path: Path) -> str:
    """标题 = 文件名去掉末尾括号及其内容，如 ...批复(FBM-CLI.3.175961) -> ...批复。"""
    stem = path.stem
    title = re.sub(r"[（(][^（）()]*[）)]\s*$", "", stem).strip()
    return title or stem


def relative_source_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def clean_text(text: str) -> str:
    """清洗：统一换行 → 去掉首尾空行 → 第1/2行相同则去重 → "此复"及其后删除。"""
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


def _hard_cut(text: str, max_chars: int) -> list[str]:
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def split_recursive(text: str, max_chars: int = MAX_CHILD_CHARS) -> list[str]:
    """递归切分：段落边界 → 句边界 → 硬切，每块 <= max_chars。"""
    text = text.strip()
    if not text:
        return []
    if norm_len(text) <= max_chars:
        return [text]

    # 1) 段落边界：空行，再换行
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

    # 2) 句边界：选最接近中点、且两半都 >= MIN_CHILD 的句末标点处切开
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

    # 3) 硬切
    return _hard_cut(text, max_chars)


def overlap_prefix(prev_text: str) -> str:
    """取上一块末尾 30–80 字作为下一块的前文上下文。"""
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
    """给后续块加前文 overlap（首块不动）。"""
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


def merge_short_chunks(chunks: list[str], min_chars: int = MIN_CHILD_CHARS) -> list[str]:
    """把 < min_chars 的短块并入前一块（最后一块向前并）。"""
    if len(chunks) <= 1:
        return chunks
    merged: list[str] = []
    buf = ""
    for ch in chunks:
        if not buf:
            buf = ch
        elif norm_len(buf) < min_chars:
            buf = buf + "\n" + ch
        else:
            merged.append(buf)
            buf = ch
    if buf:
        if merged and norm_len(buf) < min_chars:
            merged[-1] = merged[-1] + "\n" + buf
        else:
            merged.append(buf)
    return merged


def build_document(path: Path, doc_id: str) -> dict:
    raw = read_text(path)
    text = clean_text(raw)
    title = title_from_filename(path)

    child_texts = split_recursive(text)
    child_texts = add_overlap(child_texts)
    child_texts = merge_short_chunks(child_texts)

    parent_id = f"{doc_id}_p001"
    children = []
    child_ids = []
    for i, ct in enumerate(child_texts, start=1):
        cid = f"{doc_id}_c{i:03d}"
        child_ids.append(cid)
        children.append({"chunk_id": cid, "parent_id": parent_id, "title_path": title, "text": ct})

    # 单一父块 = 整篇文档（父块全文不含 overlap）
    parents = [{"parent_id": parent_id, "title_path": title, "text": text, "child_ids": child_ids}]

    return {
        "doc_id": doc_id,
        "source_path": relative_source_path(path),
        "title": title,
        "parents": parents,
        "children": children,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="切分 中央法规/1自然段文本类 为父子块 JSON。")
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
        n_children = len(data["children"])
        n_parents = len(data["parents"])
        print(f"[OK] {path.name[:40]}... -> {out.name[:40]}... (parents={n_parents}, children={n_children})")
        ok += 1

    print(f"done: ok={ok}, fail={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
