from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_JSON_DIR = ROOT_DIR / "knowledge_data" / "json"
DEFAULT_CSV_DIR = ROOT_DIR / "knowledge_data" / "embedding_csv"
DEFAULT_CONFIG_PATH = ROOT_DIR / "api_key" / "api_config.yaml"
DEFAULT_COLLECTION = "law_statutes"

VECTOR_SIZE = 1024
DISTANCE = "Cosine"
TIMEOUT_SECONDS = 30.0


def fail(message: str, exit_code: int = 1) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def quote_path(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def parse_yaml_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, dict[str, Any]]:
    config: dict[str, dict[str, Any]] = {}
    current_section: str | None = None

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())
        if indent == 0:
            if not stripped.endswith(":"):
                raise ValueError(f"Unsupported YAML line in {path}:{line_no}: {line}")
            current_section = stripped[:-1].strip()
            config[current_section] = {}
            continue

        if current_section is None or ":" not in stripped:
            raise ValueError(f"Unsupported YAML line in {path}:{line_no}: {line}")

        key, raw_value = stripped.split(":", 1)
        config[current_section][key.strip()] = parse_yaml_scalar(raw_value)

    return config


def load_api_config(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"API config file does not exist: {path}")

    try:
        import yaml
    except ImportError:
        return load_simple_yaml(path)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"API config must be a YAML object: {path}")
    return data


def load_qdrant_config(path: Path) -> tuple[str, str, str | None]:
    config = load_api_config(path)
    qdrant = config.get("qdrant")
    if not isinstance(qdrant, dict):
        raise ValueError(f"Missing `qdrant` section in: {path}")

    api_key = str(qdrant.get("api_key") or "").strip()
    url = str(qdrant.get("url") or "").strip().rstrip("/")
    collection_name = str(qdrant.get("collection_name") or "").strip() or None

    if not api_key:
        raise ValueError(f"Missing qdrant.api_key in: {path}")
    if not url:
        raise ValueError(f"Missing qdrant.url in: {path}")
    if "填" in api_key or "QDRANT_API_KEY" in api_key:
        raise ValueError(f"qdrant.api_key still looks like a placeholder in: {path}")
    if "xxxxxx" in url:
        raise ValueError(f"qdrant.url still looks like a placeholder in: {path}")
    try:
        api_key.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"qdrant.api_key contains non-ASCII/non-latin characters in: {path}"
        ) from exc
    return api_key, url, collection_name


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")


def load_children(json_dir: Path) -> dict[str, dict[str, Any]]:
    child_index: dict[str, dict[str, Any]] = {}

    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {json_dir}")

    for path in files:
        data = read_json(path)
        children = data.get("children")
        if not isinstance(children, list):
            raise ValueError(f"`children` is not a list: {path}")

        for child in children:
            if not isinstance(child, dict):
                raise ValueError(f"Child item is not an object: {path}")

            chunk_id = str(child.get("chunk_id") or "").strip()
            if not chunk_id:
                raise ValueError(f"Missing child chunk_id in: {path}")
            if chunk_id in child_index:
                raise ValueError(f"Duplicate child chunk_id in JSON files: {chunk_id}")

            child_index[chunk_id] = dict(child)

    return child_index


def parse_embedding(raw: str, source: str) -> list[float]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid embedding JSON in {source}: {exc}") from exc

    if not isinstance(value, list):
        raise ValueError(f"Embedding is not a list in {source}")
    if len(value) != VECTOR_SIZE:
        raise ValueError(f"Embedding dimension is {len(value)}, expected {VECTOR_SIZE}, in {source}")
    if not all(isinstance(item, (int, float)) for item in value):
        raise ValueError(f"Embedding contains non-numeric values in {source}")

    return [float(item) for item in value]


def load_embeddings(csv_dir: Path) -> dict[str, dict[str, Any]]:
    embedding_index: dict[str, dict[str, Any]] = {}

    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"chunk_id", "parent_id", "embedding"}
            fieldnames = set(reader.fieldnames or [])
            if not required.issubset(fieldnames):
                missing = ", ".join(sorted(required - fieldnames))
                raise ValueError(f"CSV missing required column(s) {missing}: {path}")

            for line_no, row in enumerate(reader, start=2):
                chunk_id = str(row.get("chunk_id") or "").strip()
                parent_id = str(row.get("parent_id") or "").strip()
                if not chunk_id:
                    raise ValueError(f"Missing chunk_id in {path}:{line_no}")
                if chunk_id in embedding_index:
                    raise ValueError(f"Duplicate chunk_id in CSV files: {chunk_id}")

                embedding = parse_embedding(row.get("embedding") or "", f"{path}:{line_no}")
                embedding_index[chunk_id] = {
                    "parent_id": parent_id,
                    "embedding": embedding,
                }

    return embedding_index


def point_id_from_chunk_id(chunk_id: str, mode: str) -> str:
    if mode == "chunk_id":
        return chunk_id
    if mode == "uuid5":
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agentic_rag_law:{chunk_id}"))
    raise ValueError(f"Unsupported point id mode: {mode}")


def build_points(
    child_index: dict[str, dict[str, Any]],
    embedding_index: dict[str, dict[str, Any]],
    point_id_mode: str,
) -> list[dict[str, Any]]:
    missing_embeddings = sorted(set(child_index) - set(embedding_index))
    orphan_embeddings = sorted(set(embedding_index) - set(child_index))

    if missing_embeddings:
        preview = ", ".join(missing_embeddings[:10])
        raise ValueError(f"{len(missing_embeddings)} JSON children have no embedding CSV row. Examples: {preview}")
    if orphan_embeddings:
        preview = ", ".join(orphan_embeddings[:10])
        raise ValueError(f"{len(orphan_embeddings)} CSV rows have no matching JSON child. Examples: {preview}")

    points = []
    for chunk_id in sorted(child_index):
        child = child_index[chunk_id]
        embedding_row = embedding_index[chunk_id]

        json_parent_id = str(child.get("parent_id") or "").strip()
        csv_parent_id = str(embedding_row.get("parent_id") or "").strip()
        if json_parent_id != csv_parent_id:
            raise ValueError(
                f"parent_id mismatch for {chunk_id}: JSON={json_parent_id}, CSV={csv_parent_id}"
            )

        points.append(
            {
                "id": point_id_from_chunk_id(chunk_id, point_id_mode),
                "vector": embedding_row["embedding"],
                "payload": dict(child),
            }
        )

    return points


class QdrantRestClient:
    def __init__(self, url: str, api_key: str, timeout: float = TIMEOUT_SECONDS) -> None:
        self.base_url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=data,
            method=method,
            headers={
                "api-key": self.api_key,
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qdrant HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot connect to Qdrant: {exc.reason}") from exc

        if not payload:
            return {}
        return json.loads(payload)

    def list_collections(self) -> list[str]:
        response = self.request("GET", "/collections")
        collections = response.get("result", {}).get("collections", [])
        return [item["name"] for item in collections]

    def create_collection(self, collection_name: str) -> None:
        self.request(
            "PUT",
            f"/collections/{quote_path(collection_name)}",
            {
                "vectors": {
                    "size": VECTOR_SIZE,
                    "distance": DISTANCE,
                }
            },
        )

    def delete_collection(self, collection_name: str) -> None:
        self.request("DELETE", f"/collections/{quote_path(collection_name)}")

    def upsert_points(self, collection_name: str, points: list[dict[str, Any]]) -> None:
        self.request(
            "PUT",
            f"/collections/{quote_path(collection_name)}/points?wait=true",
            {"points": points},
        )


def chunks(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload child chunk embeddings from JSON + CSV into Qdrant."
    )
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--collection",
        default=None,
        help=(
            "Qdrant collection name. Defaults to qdrant.collection_name in "
            "api_config.yaml, then falls back to the script default."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before uploading.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and build points without connecting to Qdrant.",
    )
    parser.add_argument(
        "--point-id-mode",
        choices=("chunk_id", "uuid5"),
        default="uuid5",
        help=(
            "Use deterministic UUIDv5 generated from chunk_id as point id, "
            "or use chunk_id directly. Qdrant Cloud usually requires uuid5."
        ),
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep after each successful batch upload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        fail("--batch-size must be positive")

    json_dir = args.json_dir.resolve()
    csv_dir = args.csv_dir.resolve()
    config_path = args.config.resolve()
    collection = args.collection
    qdrant_api_key = None
    qdrant_url = None

    if not args.dry_run or collection is None:
        qdrant_api_key, qdrant_url, config_collection = load_qdrant_config(config_path)
        collection = collection or config_collection
    collection = collection or DEFAULT_COLLECTION

    print(f"json_dir: {json_dir}")
    print(f"csv_dir: {csv_dir}")
    print(f"config: {config_path}")
    print(f"collection: {collection}")
    print(f"vector_size: {VECTOR_SIZE}")
    print(f"distance: {DISTANCE}")
    print(f"point_id_mode: {args.point_id_mode}")

    child_index = load_children(json_dir)
    embedding_index = load_embeddings(csv_dir)
    points = build_points(child_index, embedding_index, args.point_id_mode)

    print(f"json children: {len(child_index)}")
    print(f"csv embeddings: {len(embedding_index)}")
    print(f"points ready: {len(points)}")

    if args.dry_run:
        print("dry run complete: no Qdrant request sent")
        return 0

    if not qdrant_api_key or not qdrant_url:
        qdrant_api_key, qdrant_url, _ = load_qdrant_config(config_path)
    client = QdrantRestClient(url=qdrant_url, api_key=qdrant_api_key)

    collections = client.list_collections()
    if args.recreate and collection in collections:
        print(f"deleting existing collection: {collection}")
        client.delete_collection(collection)
        collections = [name for name in collections if name != collection]

    if collection not in collections:
        print(f"creating collection: {collection}")
        client.create_collection(collection)
    else:
        print(f"collection exists: {collection}")

    uploaded = 0
    for start, batch in chunks(points, args.batch_size):
        client.upsert_points(collection, batch)
        uploaded += len(batch)
        print(f"uploaded {uploaded}/{len(points)}")
        if args.request_sleep > 0 and start + len(batch) < len(points):
            time.sleep(args.request_sleep)

    print(f"done: uploaded {uploaded} points")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        fail(str(exc))
