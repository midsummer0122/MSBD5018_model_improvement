"""测试 Qdrant Cloud 连接、写入测试向量、检索测试向量、删除测试数据。

使用方式：
1. 在下面的 QDRANT_URL 和 QDRANT_API_KEY 中填入你的 Qdrant Cloud 信息。
2. 运行：python scripts/test_qdrant_connection.py
3. 程序会先连接 Qdrant 并列出当前 collections。
4. 输入数字 1：创建临时 collection，并写入 3 条测试向量。
5. 输入数字 2：用一条新的 query 向量做检索，并打印返回结果。
6. 输入数字 3：删除临时 collection，然后结束程序。
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any


# 这里填写你的 Qdrant Cloud 地址。
# 注意：如果官方控制台复制出来的地址没有端口，可以手动加上 :6333。
QDRANT_URL = "https://8d011e3c-6b09-4ae4-9fb7-dab107c11d0a.eu-west-2-0.aws.cloud.qdrant.io"

# 这里填写你的 Qdrant API Key。
# 注意：不要把包含真实 API Key 的代码提交到公开仓库。
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZjU4Nm" \
                 "YyODItYTg0MC00MzViLWI2MzctYTBhNDBmODdhMDUxIn0.qQ_L9pTg_4cnNGCA25hBS1UZtj2zjAeKcB57yy3k3QA"

# 请求超时时间，单位是秒。网络慢时可以适当调大。
TIMEOUT_SECONDS = 10.0

# 这是 3 条假的 4 维测试向量，只用于验证 Qdrant 的写入和检索功能。
# 正式 RAG 项目中，向量维度要和你选择的 embedding 模型一致。
TEST_POINTS = [
    {
        "id": 1,
        "vector": [0.90, 0.10, 0.10, 0.10],
        "payload": {
            "title": "劳动合同法",
            "content": "劳动合同法规定经济补偿按劳动者在本单位工作的年限计算。",
            "source_type": "statute_test",
        },
    },
    {
        "id": 2,
        "vector": [0.10, 0.90, 0.10, 0.10],
        "payload": {
            "title": "劳动争议调解仲裁法",
            "content": "劳动争议申请仲裁的时效期间为一年。",
            "source_type": "statute_test",
        },
    },
    {
        "id": 3,
        "vector": [0.10, 0.10, 0.90, 0.10],
        "payload": {
            "title": "劳动法",
            "content": "劳动法规定劳动者享有平等就业和选择职业的权利。",
            "source_type": "statute_test",
        },
    },
]

# 这是一条新的 query 向量，故意设计得更接近第 1 条测试向量。
# 所以正常情况下，检索结果里”劳动合同法”应该排在前面。
QUERY_VECTOR = [0.86, 0.18, 0.12, 0.10]


def mask_secret(value: str | None) -> str:
    """打印 API Key 时做脱敏，避免在终端里直接暴露完整密钥。"""
    if not value:
        return "<missing>"
    if len(value) <= 10:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


def fail(message: str, exit_code: int = 1) -> None:
    """统一打印失败信息并退出程序。"""
    print(f"[失败] {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def quote_path(value: str) -> str:
    """collection 名会出现在 URL 路径里，所以这里做一次安全编码。"""
    return urllib.parse.quote(value, safe="")


class QdrantRestClient:
    """一个很小的 Qdrant REST 客户端，只保留本次测试需要的能力。"""

    def __init__(self, url: str, api_key: str, timeout: float) -> None:
        self.base_url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """统一发送 HTTP 请求，连接测试、写入、读取、删除都会走这里。"""
        url = f"{self.base_url}{path}"
        data = None if body is None else json.dumps(body).encode("utf-8")
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
            fail(f"Qdrant 返回 HTTP {exc.code}：{detail}")
        except urllib.error.URLError as exc:
            fail(f"无法连接 Qdrant：{exc.reason}")

        if not payload:
            return {}
        return json.loads(payload)

    def list_collections(self) -> list[str]:
        """测试连接和鉴权：能列出 collections，就说明 URL、API Key、网络基本正常。"""
        response = self.request("GET", "/collections")
        collections = response.get("result", {}).get("collections", [])
        return [collection["name"] for collection in collections]

    def create_collection(self, collection_name: str, vector_size: int) -> None:
        """创建临时 collection，用来存放本次测试向量。"""
        self.request(
            "PUT",
            f"/collections/{quote_path(collection_name)}",
            {
                "vectors": {
                    "size": vector_size,
                    "distance": "Cosine",
                }
            },
        )

    def upsert_points(
        self,
        collection_name: str,
        points: list[dict[str, Any]],
    ) -> None:
        """写入多条测试向量和 payload，验证向量入库能力。"""
        self.request(
            "PUT",
            f"/collections/{quote_path(collection_name)}/points?wait=true",
            {
                "points": [
                    {
                        "id": point["id"],
                        "vector": point["vector"],
                        "payload": point["payload"],
                    }
                    for point in points
                ]
            },
        )

    def retrieve_points(
        self,
        collection_name: str,
        point_ids: list[int],
    ) -> list[dict[str, Any]]:
        """按 ID 读取刚写入的点，验证点查询能力。"""
        response = self.request(
            "POST",
            f"/collections/{quote_path(collection_name)}/points",
            {
                "ids": point_ids,
                "with_payload": True,
                "with_vector": False,
            },
        )
        return response.get("result", [])

    def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """用 query 向量做相似度检索，验证最基础的向量检索链路。"""
        response = self.request(
            "POST",
            f"/collections/{quote_path(collection_name)}/points/search",
            {
                "vector": vector,
                "limit": limit,
                "with_payload": True,
            },
        )
        return response.get("result", [])

    def delete_collection(self, collection_name: str) -> None:
        """删除临时 collection，避免云端留下测试用空集合。"""
        self.request("DELETE", f"/collections/{quote_path(collection_name)}")


def check_config() -> None:
    """检查脚本顶部的 Qdrant 配置是否已经填写。"""
    if not QDRANT_URL or "你的-qdrant-cloud-url" in QDRANT_URL:
        fail("请先在脚本顶部填写 QDRANT_URL。")
    if not QDRANT_API_KEY or "你的-api-key" in QDRANT_API_KEY:
        fail("请先在脚本顶部填写 QDRANT_API_KEY。")


def connect_and_show_collections() -> QdrantRestClient:
    """连接 Qdrant Cloud，并列出当前 collections。"""
    print(f"[信息] Qdrant URL: {QDRANT_URL}")
    print(f"[信息] API Key: {mask_secret(QDRANT_API_KEY)}")

    client = QdrantRestClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=TIMEOUT_SECONDS,
    )

    collections = client.list_collections()
    print("[正常] 已连接到 Qdrant Cloud。")
    print(f"[正常] 当前 collections 数量：{len(collections)}")
    for name in collections:
        print(f"  - {name}")

    return client


def write_test_vectors(client: QdrantRestClient) -> str:
    """创建临时 collection，写入 3 条测试向量，并验证读取。"""
    collection_name = f"agentic_rag_law_smoke_{uuid.uuid4().hex[:8]}"

    print(f"[信息] 创建临时 collection：{collection_name}")
    client.create_collection(collection_name, vector_size=len(TEST_POINTS[0]["vector"]))

    print("[信息] 写入 3 条测试向量。")
    client.upsert_points(collection_name=collection_name, points=TEST_POINTS)

    print("[信息] 按 ID 读取刚写入的 3 条测试向量。")
    point_ids = [point["id"] for point in TEST_POINTS]
    points = client.retrieve_points(collection_name, point_ids)
    if len(points) != len(TEST_POINTS):
        fail(f"应该读回 {len(TEST_POINTS)} 条测试向量，实际读回 {len(points)} 条。")

    print(f"[正常] 3 条测试向量写入成功，collection 名称：{collection_name}")
    return collection_name


def search_test_vectors(client: QdrantRestClient, collection_name: str | None) -> None:
    """使用新的 query 向量检索 3 条测试向量，并打印结果。"""
    if collection_name is None:
        print("[提示] 当前还没有写入测试向量。请先输入 1。")
        return

    print("[信息] 使用新的 query 向量做相似度检索。")
    print(f"[信息] Query vector: {QUERY_VECTOR}")
    search_result = client.search(collection_name, QUERY_VECTOR, limit=3)
    if not search_result:
        fail("向量检索没有返回结果。")

    print(f"[正常] 检索返回 {len(search_result)} 条结果：")
    for index, item in enumerate(search_result, start=1):
        payload = item.get("payload", {})
        print(f"  {index}. id={item.get('id')} score={item.get('score')}")
        print(f"     title={payload.get('title')}")
        print(f"     content={payload.get('content')}")


def delete_test_collection_and_close(
    client: QdrantRestClient,
    collection_name: str | None,
) -> None:
    """删除临时 collection，并结束程序。"""
    if collection_name is None:
        print("[信息] 当前还没有写入测试向量，不需要删除 collection。")
        print("[信息] REST 请求没有持久连接需要手动关闭，程序即将结束。")
        return

    print(f"[信息] 删除临时 collection：{collection_name}")
    client.delete_collection(collection_name)

    print("[正常] 临时 collection 和其中的 3 条测试向量已删除。")
    print("[信息] REST 请求没有持久连接需要手动关闭，程序即将结束。")


def main() -> None:
    check_config()
    client = connect_and_show_collections()
    test_collection_name: str | None = None

    while True:
        print()
        print("请输入操作数字：")
        print("1 = 写入 3 条测试向量")
        print("2 = 用新向量检索并返回结果")
        print("3 = 删除测试数据并关闭连接")
        choice = input("你的选择：").strip()

        if choice == "1":
            if test_collection_name is not None:
                print(f"[提示] 已经写入过测试向量：{test_collection_name}")
                print("[提示] 请继续输入 2 检索，或输入 3 删除并结束。")
                continue
            test_collection_name = write_test_vectors(client)
        elif choice == "2":
            search_test_vectors(client, test_collection_name)
        elif choice == "3":
            delete_test_collection_and_close(client, test_collection_name)
            break
        else:
            print("[提示] 请输入数字 1、2 或 3。")


if __name__ == "__main__":
    main()
