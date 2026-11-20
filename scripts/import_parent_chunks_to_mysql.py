from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_JSON_DIR = ROOT_DIR / "knowledge_data" / "json"
DEFAULT_DATABASE = "law_rag"


def fail(message: str, exit_code: int = 1) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_parent_rows(json_dir: Path, limit: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = sorted(json_dir.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {json_dir}")

    documents: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    seen_parent_ids: set[str] = set()

    for path in files:
        data = read_json(path)

        doc_id = clean_str(data.get("doc_id"))
        if not doc_id:
            raise ValueError(f"Missing doc_id in: {path}")
        if doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate doc_id in JSON files: {doc_id}")
        seen_doc_ids.add(doc_id)

        parents = data.get("parents")
        if not isinstance(parents, list):
            raise ValueError(f"`parents` is not a list: {path}")

        document = {
            "doc_id": doc_id,
            "title": clean_str(data.get("title")),
            "source_path": clean_str(data.get("source_path")),
            "source_json": path.name,
            "structure_type": clean_str(data.get("structure_type")),
            "file_hash": sha256_file(path),
        }
        documents.append(document)

        for parent_index, parent in enumerate(parents):
            if not isinstance(parent, dict):
                raise ValueError(f"Parent item is not an object: {path}")

            parent_id = clean_str(parent.get("parent_id"))
            if not parent_id:
                raise ValueError(f"Missing parent_id in: {path}")
            if parent_id in seen_parent_ids:
                raise ValueError(f"Duplicate parent_id in JSON files: {parent_id}")
            seen_parent_ids.add(parent_id)

            text = clean_str(parent.get("text"))
            if not text:
                raise ValueError(f"Missing parent text in: {path}, parent_id={parent_id}")

            child_ids = parent.get("child_ids")
            if child_ids is not None and not isinstance(child_ids, list):
                raise ValueError(f"child_ids is not a list: {path}, parent_id={parent_id}")

            parent_rows.append(
                {
                    "doc_id": doc_id,
                    "parent_id": parent_id,
                    "title_path": clean_str(parent.get("title_path")),
                    "text": text,
                    "child_ids": None
                    if child_ids is None
                    else json.dumps(child_ids, ensure_ascii=False),
                    "parent_index": parent_index,
                    "content_hash": sha256_text(text),
                }
            )

        if "children" in data:
            children = data["children"]
            if not isinstance(children, list):
                raise ValueError(f"children is not a list: {path.name}")
            child_map = {str(c.get("chunk_id") or ""): c for c in children}
            local_parents = {str(p["parent_id"]): p for p in parents}
            if len(child_map) != len(children) or "" in child_map:
                raise ValueError(f"Duplicate or empty chunk_id: {path.name}")
            for cid, child in child_map.items():
                pid = str(child.get("parent_id") or "")
                if pid not in local_parents or cid not in (local_parents[pid].get("child_ids") or []):
                    raise ValueError(f"Broken child-parent link: {cid}")
            for parent in parents:
                for cid in parent.get("child_ids") or []:
                    if cid not in child_map or child_map[cid].get("parent_id") != parent["parent_id"]:
                        raise ValueError(f"Broken parent-child link: {parent['parent_id']}")

    return documents, parent_rows


def import_pymysql():
    try:
        import pymysql
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: pymysql. Install it with `pip install pymysql`, "
            "then rerun this script."
        ) from exc
    return pymysql


def connect_mysql(args: argparse.Namespace):
    pymysql = import_pymysql()
    password = args.password
    if args.password_env:
        password = os.environ.get(args.password_env)
    if args.ask_password:
        password = getpass.getpass("MySQL password: ")

    return pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=password or "",
        database=args.database,
        charset="utf8mb4",
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )


def upsert_document(cursor: Any, document: dict[str, Any]) -> int:
    cursor.execute(
        """
        INSERT INTO documents (
          doc_id, title, source_path, source_json, structure_type, file_hash
        ) VALUES (
          %(doc_id)s, %(title)s, %(source_path)s, %(source_json)s, %(structure_type)s, %(file_hash)s
        )
        ON DUPLICATE KEY UPDATE
          id = LAST_INSERT_ID(id),
          title = VALUES(title),
          source_path = VALUES(source_path),
          source_json = VALUES(source_json),
          structure_type = VALUES(structure_type),
          file_hash = VALUES(file_hash)
        """,
        document,
    )
    return int(cursor.lastrowid)


def upsert_parent(cursor: Any, row: dict[str, Any], document_id: int) -> None:
    payload = dict(row)
    payload["document_id"] = document_id
    cursor.execute(
        """
        INSERT INTO parent_chunks (
          document_id, parent_id, title_path, text, child_ids, parent_index, content_hash
        ) VALUES (
          %(document_id)s, %(parent_id)s, %(title_path)s, %(text)s,
          %(child_ids)s, %(parent_index)s, %(content_hash)s
        )
        ON DUPLICATE KEY UPDATE
          document_id = VALUES(document_id),
          title_path = VALUES(title_path),
          text = VALUES(text),
          child_ids = VALUES(child_ids),
          parent_index = VALUES(parent_index),
          content_hash = VALUES(content_hash)
        """,
        payload,
    )


def check_existing_rows(conn: Any, documents: list[dict[str, Any]], parent_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Reject ID/content conflicts before performing any upsert; never print body text or passwords."""
    conflicts = []
    matched_documents = matched_parents = 0
    with conn.cursor() as cursor:
        for start in range(0, len(documents), 300):
            batch = documents[start:start+300]
            placeholders = ",".join(["%s"] * len(batch))
            cursor.execute(f"SELECT doc_id,title,source_path,source_json,structure_type FROM documents WHERE doc_id IN ({placeholders})",
                           [d["doc_id"] for d in batch])
            existing = {str(row["doc_id"]): row for row in cursor.fetchall()}
            for document in batch:
                old = existing.get(document["doc_id"])
                if old is None: continue
                matched_documents += 1
                for key in ("title", "source_path", "source_json", "structure_type"):
                    if clean_str(old.get(key)) != document.get(key):
                        conflicts.append(f"document {document['doc_id']}: {key}")
        for start in range(0, len(parent_rows), 300):
            batch = parent_rows[start:start+300]
            placeholders = ",".join(["%s"] * len(batch))
            cursor.execute(f"""SELECT pc.parent_id,pc.title_path,pc.text,pc.child_ids,pc.parent_index,
                               pc.content_hash,d.doc_id FROM parent_chunks pc
                               LEFT JOIN documents d ON d.id=pc.document_id
                               WHERE pc.parent_id IN ({placeholders})""", [p["parent_id"] for p in batch])
            existing = {str(row["parent_id"]): row for row in cursor.fetchall()}
            for parent in batch:
                old = existing.get(parent["parent_id"])
                if old is None: continue
                matched_parents += 1
                for key in ("doc_id", "title_path", "text", "parent_index"):
                    if old.get(key) != parent.get(key):
                        conflicts.append(f"parent {parent['parent_id']}: {key}")
                old_ids = old.get("child_ids")
                if isinstance(old_ids, str): old_ids = json.loads(old_ids)
                new_ids = json.loads(parent["child_ids"]) if parent["child_ids"] is not None else None
                if old_ids != new_ids:
                    conflicts.append(f"parent {parent['parent_id']}: child_ids")
                if old.get("content_hash") != parent["content_hash"]:
                    conflicts.append(f"parent {parent['parent_id']}: content_hash")
    if conflicts:
        raise ValueError(f"ID/content conflicts detected; no import started. Count={len(conflicts)}; " + "; ".join(conflicts[:12]))
    return {"matching_documents": matched_documents, "matching_parents": matched_parents}


def write_to_mysql(args: argparse.Namespace, documents: list[dict[str, Any]], parent_rows: list[dict[str, Any]]) -> None:
    parents_by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in parent_rows:
        parents_by_doc.setdefault(row["doc_id"], []).append(row)

    conn = connect_mysql(args)
    try:
        checked = check_existing_rows(conn, documents, parent_rows)
        print(f"conflict check passed: matching_documents={checked['matching_documents']}, matching_parents={checked['matching_parents']}")
        with conn.cursor() as cursor:
            imported_documents = 0
            imported_parents = 0

            for document in documents:
                document_id = upsert_document(cursor, document)
                imported_documents += 1

                for parent in parents_by_doc.get(document["doc_id"], []):
                    upsert_parent(cursor, parent, document_id)
                    imported_parents += 1

                if imported_documents % args.commit_every == 0:
                    conn.commit()
                    print(
                        f"committed documents={imported_documents}/{len(documents)}, "
                        f"parents={imported_parents}/{len(parent_rows)}"
                    )

            conn.commit()
            print(f"done: documents={imported_documents}, parent_chunks={imported_parents}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import JSON parent chunks into MySQL documents and parent_chunks tables."
    )
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--host", default=os.environ.get("MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MYSQL_PORT", "3306")))
    parser.add_argument("--user", default=os.environ.get("MYSQL_USER", "root"))
    parser.add_argument("--password", default=os.environ.get("MYSQL_PASSWORD"))
    parser.add_argument(
        "--password-env",
        default=None,
        help="Read MySQL password from the named environment variable.",
    )
    parser.add_argument(
        "--ask-password",
        action="store_true",
        help="Prompt for MySQL password instead of passing it on the command line.",
    )
    parser.add_argument("--database", default=os.environ.get("MYSQL_DATABASE", DEFAULT_DATABASE))
    parser.add_argument("--commit-every", type=int, default=20)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only load the first N JSON files. Useful for testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate JSON and print counts without connecting to MySQL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.commit_every <= 0:
        fail("--commit-every must be positive")
    if args.limit is not None and args.limit <= 0:
        fail("--limit must be positive")

    json_dir = args.json_dir.resolve()
    documents, parent_rows = load_parent_rows(json_dir=json_dir, limit=args.limit)

    print(f"json_dir: {json_dir}")
    print(f"database: {args.database}")
    print(f"documents ready: {len(documents)}")
    print(f"parent chunks ready: {len(parent_rows)}")

    if args.dry_run:
        print("dry run complete: no MySQL writes")
        return 0

    write_to_mysql(args, documents, parent_rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        fail(str(exc))
