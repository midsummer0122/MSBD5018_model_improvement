import argparse
import csv
import json
import time
from pathlib import Path

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_JSON_DIR = ROOT_DIR / "knowledge_data" / "json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "knowledge_data" / "embedding_csv"
DEFAULT_CONFIG_PATH = ROOT_DIR / "api_key" / "api_config.yaml"

MODEL = "text-embedding-v4"
DIMENSIONS = 1024
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def parse_yaml_scalar(raw: str):
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


def load_simple_yaml(path: Path) -> dict:
    config = {}
    current_section = None

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


def load_api_config(path: Path) -> dict:
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


def load_embedding_api_key(path: Path) -> str:
    config = load_api_config(path)
    embedding = config.get("embedding")
    if not isinstance(embedding, dict):
        raise ValueError(f"Missing `embedding` section in: {path}")

    api_key = str(embedding.get("api_key") or "").strip()
    if not api_key:
        raise ValueError(f"Missing embedding.api_key in: {path}")
    return api_key


def read_json(path: Path) -> dict:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")


def build_embedding_text(child: dict) -> str:
    title_path = str(child.get("title_path") or "").strip()
    body = str(child.get("text") or "").strip()

    parts = []
    if title_path:
        parts.append(f"标题路径：{title_path}")
    if body:
        parts.append(f"正文：{body}")
    return "\n".join(parts).strip()


def create_embeddings(client: OpenAI, texts: list[str], retries: int, retry_sleep: float) -> list[list[float]]:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            completion = client.embeddings.create(
                model=MODEL,
                input=texts,
                dimensions=DIMENSIONS,
                encoding_format="float",
            )
            items = sorted(completion.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(texts))):
                raise ValueError("Embedding response indices do not match input batch")
            embeddings = [item.embedding for item in items]
            if any(len(vector) != DIMENSIONS for vector in embeddings):
                raise ValueError("Unexpected embedding dimensions")
            return embeddings
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"Embedding request failed after {retries} attempts") from last_error


def load_existing_chunk_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    existing_ids = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "chunk_id" not in reader.fieldnames:
            return existing_ids

        for row in reader:
            chunk_id = (row.get("chunk_id") or "").strip()
            embedding_text = row.get("embedding") or ""
            try:
                embedding = json.loads(embedding_text)
            except json.JSONDecodeError:
                continue
            if (
                chunk_id
                and isinstance(embedding, list)
                and len(embedding) == DIMENSIONS
                and all(isinstance(x, (int, float)) for x in embedding)
            ):
                existing_ids.add(chunk_id)
    return existing_ids


def ensure_csv_header(csv_path: Path) -> None:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chunk_id", "parent_id", "embedding"],
        )
        writer.writeheader()


def append_csv_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    ensure_csv_header(csv_path)
    with csv_path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chunk_id", "parent_id", "embedding"],
        )
        writer.writerows(rows)


def embed_json_to_csv(
    path: Path,
    csv_path: Path,
    client: OpenAI,
    overwrite: bool,
    retries: int,
    retry_sleep: float,
    request_sleep: float,
    batch_size: int = 1,
) -> tuple[int, int]:
    data = read_json(path)
    children = data.get("children")
    if not isinstance(children, list):
        raise ValueError(f"`children` is not a list: {path}")

    if overwrite and csv_path.exists():
        csv_path.unlink()

    existing_chunk_ids = set() if overwrite else load_existing_chunk_ids(csv_path)
    ensure_csv_header(csv_path)

    added_count = 0
    skipped_count = 0
    pending = []
    for child in children:
        chunk_id = str(child.get("chunk_id") or "").strip()
        if not chunk_id:
            raise ValueError(f"Missing chunk_id in {path.name}")

        if chunk_id in existing_chunk_ids:
            skipped_count += 1
            continue

        embedding_text = build_embedding_text(child)
        if not embedding_text:
            raise ValueError(f"Empty embedding text in {path.name}, chunk_id={chunk_id}")

        pending.append((child, embedding_text))

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        embeddings = create_embeddings(client, [text for _, text in batch], retries, retry_sleep)
        rows = [{"chunk_id": child["chunk_id"], "parent_id": child.get("parent_id", ""),
                 "embedding": json.dumps(vector, ensure_ascii=False, separators=(",", ":"))}
                for (child, _), vector in zip(batch, embeddings)]
        append_csv_rows(csv_path, rows)
        added_count += len(rows)

        if request_sleep > 0:
            time.sleep(request_sleep)

    return added_count, skipped_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Aliyun text-embedding-v4 vectors to each child chunk in JSON files."
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Texts per API request (1-10 for text-embedding-v4). Completed batches are resumable.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=DEFAULT_JSON_DIR,
        help=f"Directory containing chunk JSON files. Default: {DEFAULT_JSON_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output CSV files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config file containing embedding.api_key. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute embeddings and replace existing CSV files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N JSON files. Use 0 for all files.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for each embedding request.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=2.0,
        help="Base seconds to sleep between retries.",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep after each successful request.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.batch_size <= 10:
        raise ValueError("--batch-size must be between 1 and 10")
    json_dir = args.json_dir.resolve()
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory does not exist: {json_dir}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config.resolve()

    files = sorted(json_dir.glob("*.json"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {json_dir}")

    api_key = load_embedding_api_key(config_path)
    embedding_config = load_api_config(config_path).get("embedding") or {}
    global MODEL, DIMENSIONS
    MODEL = str(embedding_config.get("model") or MODEL)
    DIMENSIONS = int(embedding_config.get("dimension") or DIMENSIONS)
    client = OpenAI(api_key=api_key, base_url=str(embedding_config.get("base_url") or BASE_URL),
                    timeout=float(embedding_config.get("timeout_seconds") or 30), max_retries=0)

    total_added = 0
    total_skipped = 0

    print(f"model: {MODEL}")
    print(f"dimensions: {DIMENSIONS}")
    print(f"json_dir: {json_dir}")
    print(f"output_dir: {output_dir}")
    print(f"config: {config_path}")
    print(f"files: {len(files)}")

    for index, path in enumerate(files, start=1):
        csv_path = output_dir / f"{path.stem}.csv"
        added, skipped = embed_json_to_csv(
            path=path,
            csv_path=csv_path,
            client=client,
            overwrite=args.overwrite,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            request_sleep=args.request_sleep,
            batch_size=args.batch_size,
        )
        total_added += added
        total_skipped += skipped
        print(
            f"[{index}/{len(files)}] {path.name} "
            f"-> {csv_path.name} added={added} skipped={skipped} total_added={total_added}"
        )

    print(f"done: added={total_added}, skipped={total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
