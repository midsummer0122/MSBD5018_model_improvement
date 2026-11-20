import json
import re
import urllib.error
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
API_KEY_PATH = ROOT_DIR / "api_key" / "ali_api.txt"

MODEL = "qwen3-rerank"
RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"

QUERY = "劳动合同解除的经济补偿如何计算？"
DOCUMENTS = [
    "用人单位依法解除劳动合同的，应当向劳动者支付经济补偿。经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。",
    "劳动合同法第四十六条规定了用人单位应当向劳动者支付经济补偿的情形，包括劳动者依照第三十八条解除劳动合同、用人单位依照第四十条解除等。",
    "经济补偿的月工资是指劳动者在劳动合同解除或者终止前十二个月的平均工资。",
    "法律文本有大量专有名词、法条引用、案由、时效等表达，语义细节比普通 FAQ 更复杂，直接从 1024 起步比较合理。",
    "劳动争议申请仲裁的时效期间为一年，仲裁时效期间从当事人知道或者应当知道其权利被侵害之日起计算。",
    "劳动争议调解仲裁法规定了劳动争议的处理程序，包括协商、调解、仲裁、诉讼等环节。",
]


def load_api_key(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    match = re.search(r"sk-[A-Za-z0-9_-]+", content)
    if not match:
        raise ValueError(f"API key file is empty: {path}")
    return match.group(0)


def rerank(api_key: str) -> dict:
    payload = {
        "model": MODEL,
        "query": QUERY,
        "documents": DOCUMENTS,
        "top_n": 6,
        "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        RERANK_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ali rerank request failed: HTTP {exc.code}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ali rerank request failed: {exc.reason}") from exc


def main() -> int:
    result = rerank(load_api_key(API_KEY_PATH))
    ranked_results = result.get("output", {}).get("results", result.get("results", []))

    print(f"model: {MODEL}")
    print(f"query: {QUERY}")
    print("top_n: 6")
    print()

    for rank, item in enumerate(ranked_results, start=1):
        index = item["index"]
        score = item["relevance_score"]
        document = item.get("document", {}).get("text", DOCUMENTS[index])
        print(f"rank {rank}")
        print(f"index: {index}")
        print(f"score: {score}")
        print(f"document: {document}")
        print()

    print("raw_response:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
