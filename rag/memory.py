from __future__ import annotations

import uuid

from .schemas import ConversationState, Message


class MemoryStore:
    """进程内短期记忆。

    只保存“当前活跃话题的追问上下文”，不做磁盘持久化，程序退出即清空。
    一个话题（问题 → 追问 → 回答）结束后由 workflow 调用 ``forget`` 清除，
    因此用户下一次提问不会受之前问答记录的影响。
    """

    def __init__(self) -> None:
        self._sessions: dict[str, tuple[list[Message], ConversationState]] = {}

    def new_session_id(self) -> str:
        return f"session_{uuid.uuid4().hex[:12]}"

    def load(self, session_id: str | None) -> tuple[str, list[Message], ConversationState]:
        session_id = session_id or self.new_session_id()
        entry = self._sessions.get(session_id)
        if entry is not None:
            return session_id, entry[0], entry[1]
        return session_id, [], ConversationState(session_id=session_id)

    def save(self, session_id: str, messages: list[Message], state: ConversationState) -> None:
        self._sessions[session_id] = (messages, state)

    def forget(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
