from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .config import AppConfig


class HttpJsonClient:
    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url=url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot connect to {url}: {exc.reason}") from exc
        return {} if not payload else json.loads(payload)


class ChatProvider:
    def __init__(self, config: AppConfig) -> None:
        section = config.section("llm")
        self.base_url = str(section["base_url"]).rstrip("/")
        self.api_key = str(section["api_key"])
        self.model = str(section.get("chat_model") or "deepseek-chat")
        self.timeout = float(section.get("timeout_seconds") or 30)
        self.http = HttpJsonClient(timeout=self.timeout)

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        response = self.http.request(
            "POST",
            f"{self.base_url}/chat/completions",
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            },
        )
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat response: {response}") from exc

    def stream_chat(self, messages: list[dict[str, str]], temperature: float = 0.2):
        data = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text or not text.startswith("data:"):
                        continue
                    payload = text[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield str(content)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} for {self.base_url}/chat/completions: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot connect to {self.base_url}/chat/completions: {exc.reason}") from exc


class EmbeddingProvider:
    def __init__(self, config: AppConfig) -> None:
        section = config.section("embedding")
        self.base_url = str(section["base_url"]).rstrip("/")
        self.api_key = str(section["api_key"])
        self.model = str(section["model"])
        self.dimension = int(section.get("dimension") or 1024)
        self.timeout = float(section.get("timeout_seconds") or 30)
        self.http = HttpJsonClient(timeout=self.timeout)

    def embed(self, text: str) -> list[float]:
        response = self.http.request(
            "POST",
            f"{self.base_url}/embeddings",
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            {
                "model": self.model,
                "input": text,
                "dimensions": self.dimension,
                "encoding_format": "float",
            },
        )
        try:
            return [float(item) for item in response["data"][0]["embedding"]]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected embedding response: {response}") from exc


class RerankProvider:
    def __init__(self, config: AppConfig) -> None:
        section = config.section("rerank")
        self.url = str(section["base_url"])
        self.api_key = str(section["api_key"])
        self.model = str(section["model"])
        self.timeout = float(section.get("timeout_seconds") or 30)
        self.http = HttpJsonClient(timeout=self.timeout)

    def request_parameters(self, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
        """The exact nonsecret wire body, shared with strict experiment cache signatures."""
        return {"model": self.model, "query": query, "documents": documents,
                "top_n": min(top_n, len(documents)),
                "instruct": "Given a Chinese legal question, retrieve relevant statutes and cases that answer it."}

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict[str, Any]]:
        if not documents:
            return []
        response = self.http.request(
            "POST",
            self.url,
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            self.request_parameters(query, documents, top_n),
        )
        return response.get("output", {}).get("results", response.get("results", []))


class QdrantClient:
    def __init__(self, config: AppConfig) -> None:
        section = config.section("qdrant")
        self.url = str(section["url"]).rstrip("/")
        self.api_key = str(section["api_key"])
        self.collection = str(section["collection_name"])
        self.timeout = float(section.get("timeout_seconds") or 30)
        self.http = HttpJsonClient(timeout=self.timeout)

    @staticmethod
    def quote(value: str) -> str:
        return urllib.parse.quote(value, safe="")

    def search(self, vector: list[float], limit: int = 30, filters: dict[str, Any] | None = None, collection: str | None = None) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "vector": vector,
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        if filters:
            body["filter"] = filters
        collection = collection or self.collection
        response = self.http.request(
            "POST",
            f"{self.url}/collections/{self.quote(collection)}/points/search",
            {
                "api-key": self.api_key,
                "Content-Type": "application/json",
            },
            body,
        )
        return response.get("result", [])


class MockChatProvider:
    """离线降级用的本地聊天 provider（未配置 LLM 密钥时启用）。"""

    offline = True

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        system = ""
        if messages:
            first = messages[0]
            system = str(first.get("content") if isinstance(first, dict) else first)
        if "JSON" in system:
            # 结构化调用：返回空对象，由调用方的规则兜底/默认值接管
            return "{}"
        return (
            "【离线演示模式】当前未配置 LLM 密钥、也未接入法律知识库，本次回答为占位内容，"
            "不代表真实法律分析。\n\n"
            "完成以下配置后即可获得真实回答：\n"
            "1. 将 api_key/api_config - template.yaml 复制为 api_key/api_config.yaml，并填写 DeepSeek、"
            "阿里云、Qdrant、MySQL 的密钥；\n"
            "2. 将法条原始文本放入 knowledge_data/法条，再依次运行 "
            "scripts/chunk_legal_texts.py、scripts/embed_json_children_ali.py、"
            "scripts/import_parent_chunks_to_mysql.py、scripts/upload_child_embeddings_to_qdrant.py 构建知识库。"
        )

    def stream_chat(self, messages: list[dict[str, str]], temperature: float = 0.2):
        yield self.chat(messages, temperature)


class MockEmbeddingProvider:
    offline = True

    def embed(self, text: str) -> list[float]:
        return [0.0] * 1024


class MockRerankProvider:
    offline = True

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict[str, Any]]:
        return []


class MockQdrantClient:
    offline = True

    def search(self, vector: list[float] | None = None, limit: int = 30, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return []


class Providers:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig.load()
        self.chat = self._chat()
        self.embedding = self._embedding()
        self.rerank = self._rerank()
        self.qdrant = self._qdrant()
        self.offline = isinstance(self.chat, MockChatProvider)

    def _has_key(self, section: str, key: str) -> bool:
        try:
            sec = self.config.raw.get(section) or {}
            return bool(str(sec.get(key) or "").strip())
        except Exception:
            return False

    def _chat(self):
        return ChatProvider(self.config) if self._has_key("llm", "api_key") else MockChatProvider()

    def _embedding(self):
        return EmbeddingProvider(self.config) if self._has_key("embedding", "api_key") else MockEmbeddingProvider()

    def _rerank(self):
        return RerankProvider(self.config) if self._has_key("rerank", "api_key") else MockRerankProvider()

    def _qdrant(self):
        return QdrantClient(self.config) if self._has_key("qdrant", "api_key") else MockQdrantClient()
