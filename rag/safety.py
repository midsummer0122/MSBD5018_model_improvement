from __future__ import annotations

import json
import re

from .providers import ChatProvider
from .schemas import EvidencePack, SafetyReview, to_dict


def _extract_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def llm_safety_check(
    chat: ChatProvider,
    query: str,
    answer: str,
    evidence: EvidencePack,
) -> SafetyReview:
    prompt = f"""
你是劳动法问答系统的安全审查模块。请只输出 JSON。
请按答案实际使用的语言理解语义（支持中文、英文等多语言），不要因语言差异误判。

## 审查维度

### 1. 事实一致性
检查答案是否将用户明确提到的事实错误地写成否定，或张冠李戴（把A主体/法条/期限安到B上）。注意：允许答案在引用用户原话后给出分析判断。

### 2. 确定性表述
- 禁止对"案件结果"打包票：如"一定(能)赢""保证胜诉""包赢""稳赢""百分百能赢"等确定性承诺，应改用"可能""倾向于""根据现有证据""初步判断"等不确定措辞。
- 但允许基于法律依据的事实性确定性表述：如"根据《劳动合同法》第X条，用人单位在该情形下应当/不得……"。这种由法律规定带来的确定性结论，只要有真实法律依据，不算违规。

### 3. 假设性分类讨论
若用户对某些关键信息未明确（系统追问可能未覆盖），答案采用"如果已经……那么……；如果尚未……那么……"的分类讨论，属于对缺失信息的合理假设，不算事实错误或越界，应允许。

### 4. 是否构成正式法律意见
答案是否越界替用户做出确定性的法律结论、代替律师出具正式法律意见？应保持"仅供参考、不构成正式法律意见"的边界，必要时提示咨询执业律师。

### 5. 证据支撑
- 优先引用 Evidence Pack 中真实出现的法条，并尽量精确到条/款/项。
- 若 Evidence Pack 为空或未命中相关法条，答案可基于一般劳动法知识作答，但必须明确说明"未检索到具体法条原文，以下为一般性说明，建议核实"，不得假装引用了证据中的法条。
- 只有"编造了具体条号并声称来自证据"才驳回；证据不足但已如实说明的，不驳回。

## 输出格式
{{"approved": true/false, "issues": ["问题1", "问题2"], "rewrite_instruction": "修正指令"}}

注意：approved 为 false 时 rewrite_instruction 必填。若 approved 为 true，issues 留空数组。

## 输入

用户问题：{query}
证据摘要：{json.dumps(to_dict(evidence), ensure_ascii=False)[:4000]}
待审查答案：{answer}
""".strip()
    try:
        raw = chat.chat(
            [
                {"role": "system", "content": "你只输出可解析 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        data = _extract_json(raw)
        return SafetyReview(
            approved=bool(data.get("approved", True)),
            issues=[str(item) for item in data.get("issues", [])],
            rewrite_instruction=str(data.get("rewrite_instruction") or ""),
        )
    except Exception:
        return SafetyReview(approved=True)
