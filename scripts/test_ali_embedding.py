import json
from pathlib import Path

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
API_KEY_PATH = ROOT_DIR / "api_key" / "ali_api.txt"
TEXT_PATH = ROOT_DIR / "knowledge_data" / "test1.txt"

MODEL = "text-embedding-v4"
DIMENSIONS = 1024
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")


def load_api_key(path: Path) -> str:
    api_key = path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"API key file is empty: {path}")
    return api_key


def create_embedding(api_key: str, text: str) -> list[float]:
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    completion = client.embeddings.create(
        model=MODEL,
        input=text,
        dimensions=DIMENSIONS,
        encoding_format="float",
    )
    return completion.data[0].embedding


def main() -> int:
    api_key = load_api_key(API_KEY_PATH)
    text = read_text(TEXT_PATH)
    embedding = create_embedding(api_key, text)

    print(f"model: {MODEL}")
    print(f"source: {TEXT_PATH}")
    print(f"dimensions: {len(embedding)}")
    print(json.dumps(embedding, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
