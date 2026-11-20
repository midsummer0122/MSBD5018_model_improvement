from __future__ import annotations

import json
import re
from typing import Any

from .providers import ChatProvider
from .schemas import ConversationState, FollowupAnalysisResult, Message, QueryUnderstandingResult, TopicContinuationResult, to_dict


# 法律场景规则提示词（仅作为 LLM 无输出时的兜底，需结合上下文判断否定语义）
STATUTE_HINTS = ["法条", "条文", "条款", "规定", "构成要件", "效力", "第几条", "什么意思"]
PROCEDURE_HINTS = ["时效", "管辖", "上诉", "举证", "保全", "仲裁", "起诉", "执行", "立案", "期间", "期限"]
CONSULT_HINTS = ["怎么办", "怎么处理", "能不能", "是否违法", "维权", "赔偿", "辞退", "欠薪", "解除", "违约"]
CASE_HINTS = ["案例", "判例", "类案", "裁判", "怎么判", "判多少", "类似案件"]


def _extract_json(text: str) -> dict[str, Any]:
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


def _list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def rule_hints(query: str) -> dict[str, Any]:
    intents = []
    if any(hint in query for hint in STATUTE_HINTS):
        intents.append("法条解释")
    if any(hint in query for hint in PROCEDURE_HINTS):
        intents.append("程序咨询")
    if any(hint in query for hint in CONSULT_HINTS):
        intents.append("法律咨询")
    if any(hint in query for hint in CASE_HINTS):
        intents.append("案例检索")
    return {
        "intents": intents,
        "retrieval_targets": ["statute_db", "case_db"],
    }


def understand_query(
    chat: ChatProvider,
    query: str,
    messages: list[Message],
    state: ConversationState,
) -> QueryUnderstandingResult:
    hints = rule_hints(query)
    prompt = f"""
你是劳动法 Agentic RAG 问答系统的问题理解模块。请只输出 JSON，不要输出解释。
本项目主要面向劳动法问答（劳动合同、工资工时、解除/终止、经济补偿、工伤、社会保险、劳动争议仲裁等）。
任务：识别用户意图、实体、缺失信息、是否需要追问、检索目标、话题关系、检索改写 query。
不构成正式法律意见，不承诺胜诉。

## 语言规则
- 用户可能用中文或英文提问。
- missing_info（需要追问的问题）必须使用用户提问的语言：用户用英文提问就用英文追问，用中文就用中文追问。
- rewritten_query（检索改写）始终用中文，因为法条库是中文，即使用户用英文提问也要转成中文再检索。
- 其余字段（intents、entities、satisfied_info、user_goal 等）用中文（内部使用）。

## 字段定义

### intents（意图分类，从以下闭集中选择，可多选）
- 法律咨询：用户描述自身处境，寻求处理建议
- 法条解释：询问某法条的含义、构成要件
- 程序咨询：涉及诉讼时效、管辖、上诉、举证期限、保全等程序
- 案例检索：希望了解类似案件的裁判情况
- 其他：以上均不匹配（只要命中任一上述类别，就不输出"其他"）

### entities（实体信息，object，只输出用户明确提到的内容，未提及的字段可省略）
注意：被否定词管辖的列举项不应提取为实体。例如"没有书面合同、工资记录"中，书面合同/工资记录均被否定，不要提取。
- subjects: 主体，如 ["用人单位", "劳动者", "员工", "公司"]
- legal_relations: 劳动关系相关，如 ["劳动关系", "劳务关系", "劳务派遣", "非全日制用工"]
- amounts: 数额/标的，如 "5000元"、"三个月工资"、"经济补偿金"
- deadlines: 期限/时间节点，如 "试用期3个月"、"合同期满"、"已过2年"、"仲裁时效"
- jurisdiction: 管辖地/工作地，如 "在深圳工作"、"公司注册地在北京"
- causes: 案由（劳动争议类），如 "拖欠劳动报酬"、"违法解除劳动合同"、"工伤待遇争议"
- crimes: 罪名，如 "拒不支付劳动报酬罪"
- statutes: 劳动法规名，如 "劳动合同法"、"劳动争议调解仲裁法"、"劳动法"、"社会保险法"
- cases: 案号

### query_type（问题类型，string）
- knowledge_question: 纯法条/概念/知识类，不涉及具体个案。如"劳动合同法第四十六条讲什么""什么是诉讼时效"
- legal_consultation: 用户围绕自身或他人的具体法律处境寻求分析和建议
- procedure_consultation: 明确涉及程序性事项（时效、管辖、上诉、举证、保全、执行等）

区分要点：如果用户只是问"某某法条讲什么""什么是某概念"，属于 knowledge_question。如果用户问"我这种情况该怎么办""公司拖欠我工资怎么维权"，属于 legal_consultation。

### missing_info（需要追问的问题，string[]）与 need_followup（boolean）
missing_info 是需要向用户追问的问题，用自然语言、可直接展示给用户的形式写出（如"具体拖欠的工资金额是多少？""是否有书面劳动合同或工资记录？"）。

判断规则（必须严格遵守）：
1. 只追问用户**完全未提供**的关键信息。用户已经明确或隐含提供的信息，不要追问。
   - 隐含信息也算已提供：例如"老板欠我工资"已经隐含"我是员工/劳动者"，不要再追问当事人身份；"我公司的员工要离职"已经隐含"我是用人单位/老板"。
   - 能从上下文明确推断的身份、法律关系等信息，不要追问。
2. 否定即回答：用户说了"没有""无""不是""正常""没""否""没事""不""未"等，表示该情况不存在，已回应，不要再追问。
   特别注意中文列举管辖规则：当"没有""无"等否定词后面跟着用顿号（、）、逗号（，）或"和""与""及""或"等连接的多个项目时，否定词管辖整个列表，列表中所有项目均被否定。例如"没有书面合同、工资记录、考勤记录、转账凭证"表示全部不存在。
   反向规则同等重要：否定词仅管辖其明确列举的项。用户说"没有书面合同"只否定书面合同，不等于否定了其他证据；只有"没有书面合同、工资记录"、"没有书面合同和工资记录"、"没有书面合同等"才表示否定范围覆盖工资记录。不要把未被明确提及的追问项当作已被否定。
   此外注意中文后置否定句式：否定词在关键词之后同样表示否定。如"书面合同倒是没有""合同是没有的""工资记录的话倒没有"均表示该证据不存在。不要因为否定词在关键词后面就误认为该证据被肯定提及。
3. 描述即回答：用户给出描述性语句（如"就是普通欠薪""大概三个月""差不多那样""还行"），即使不够精确也已回应，不要重复追问。
4. need_followup 仅在存在需要追问的问题时才设为 true。用户已提供足够信息时设为 false。
5. missing_info 每项是自然语言问句，可直接展示给用户，使用与用户提问一致的语言（用户用英文提问则用英文追问，用中文则用中文），禁止过于专业的术语缩写。
6. 用户表示不理解（"不理解""看不懂""什么意思""没听懂""不太明白"）时，将让用户困惑的问题用更通俗的中文重新表述，放入 missing_info，need_followup 设为 true。
7. 禁止追问隐私信息：不要追问公司具体名称、个人姓名、身份证号、具体家庭住址、电话号码、银行账号等可识别个人或企业身份的信息。需要了解用人单位情况时，改问泛化信息（如公司类型：国企/私企/外企/个体工商户、所属行业、用工形式等）；需要了解工作地点时，只问城市/地区级别（如"在哪个城市工作"），不问具体地址。

### satisfied_info（用户已确认的关键信息点，string[]）
记录用户已确认的关键信息点，用简洁的自然语言描述，供跨话题信息延续判断使用。例如：
- "当事人身份：劳动者"
- "法律关系：劳动关系"
- "证据：无书面合同"
只要用户明确或隐含提供了某个信息点，就列入；否定也算已确认（如"没有书面合同"）。空数组表示没有确认的信息点。

### 其他字段
- retrieval_targets: string[]，检索目标，取 ["statute_db", "case_db"]（法条库与案例库）
- user_goal: string，用户的核心需求，用一句简短中文概括
- topic_relation: same_topic | related_topic | new_topic，与上文对话的主题关系
- rewritten_query: string，检索优化后的 query，去除口语化表达，保留关键法律术语。注意：rewritten_query 始终用中文（法条库为中文），即使用户用英文提问也要转成中文。

## Few-shot 示例

示例1 —— 法条解释：
用户问题：劳动合同法第四十六条讲了什么？
输出：
{{"intents": ["法条解释"], "entities": {{"statutes": ["劳动合同法"]}}, "missing_info": [], "need_followup": false, "retrieval_targets": ["statute_db"], "query_type": "knowledge_question", "user_goal": "了解劳动合同法第四十六条的规定", "topic_relation": "same_topic", "rewritten_query": "劳动合同法 第四十六条", "satisfied_info": []}}

示例2 —— 一般法律咨询：
用户问题：公司拖欠我三个月工资，我该怎么办？
输出：
{{"intents": ["法律咨询"], "entities": {{"subjects": ["公司"], "legal_relations": ["劳动关系"], "amounts": ["三个月工资"]}}, "missing_info": ["您是否已离职或仍在职？", "是否有书面劳动合同或工资记录？", "是否已向劳动监察或仲裁机构主张过？"], "need_followup": true, "retrieval_targets": ["statute_db"], "query_type": "legal_consultation", "user_goal": "了解追索被拖欠工资的途径", "topic_relation": "same_topic", "rewritten_query": "公司拖欠工资 追索 劳动仲裁", "satisfied_info": ["当事人身份：劳动者", "法律关系：劳动关系"]}}

示例3 —— 否定式描述（顿号列举否定，不应追问）：
对话历史：系统追问是否有书面劳动合同、工资记录、考勤记录、转账凭证等证据。
用户回复：没有书面合同、工资记录、考勤记录和转账凭证，都是口头约定的。
注意："没有"管辖整个顿号列举列表，书面合同/工资记录/考勤记录/转账凭证全被否定，不提取为实体，不纳入 missing_info。
输出：
{{"intents": ["法律咨询"], "entities": {{"legal_relations": ["劳动关系"]}}, "missing_info": [], "need_followup": false, "retrieval_targets": ["statute_db"], "query_type": "legal_consultation", "user_goal": "了解无书面证据时如何维权", "topic_relation": "same_topic", "rewritten_query": "无书面证据 劳动关系 维权", "satisfied_info": ["证据：无书面合同"]}}

示例4 —— 程序咨询（纯知识）：
用户问题：劳动争议申请仲裁的时效是多久？
输出：
{{"intents": ["程序咨询"], "entities": {{"legal_relations": ["劳动关系"]}}, "missing_info": [], "need_followup": false, "retrieval_targets": ["statute_db"], "query_type": "procedure_consultation", "user_goal": "了解劳动争议仲裁时效", "topic_relation": "same_topic", "rewritten_query": "劳动争议 仲裁时效", "satisfied_info": []}}

## 输入

最近上下文：
{json.dumps([to_dict(m) for m in messages[-6:]], ensure_ascii=False)}

当前会话状态：
{json.dumps(to_dict(state), ensure_ascii=False)}

关键词提示（仅供参考，注意其中关键词可能被用户否定，需结合上下文判断）：
{json.dumps(hints, ensure_ascii=False)}

用户问题：{query}
""".strip()
    try:
        raw = chat.chat(
            [
                {"role": "system", "content": "你只输出可解析 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        data = _extract_json(raw)
    except Exception:
        data = {}

    intents = _list(data.get("intents"))
    retrieval_targets = _list(data.get("retrieval_targets"))

    # Rule hints as fallback only when LLM returns nothing for a field
    if not intents:
        intents = hints["intents"]
    if not retrieval_targets:
        retrieval_targets = hints["retrieval_targets"] or ["statute_db"]

    # Enforce "其他" intent rules:
    # - "其他" must not co-exist with other intents (LLM might ignore the prompt rule)
    # - If intents is still empty after fallback, default to ["其他"]
    if len(intents) > 1 and "其他" in intents:
        intents = [i for i in intents if i != "其他"]
    if not intents:
        intents = ["其他"]

    return QueryUnderstandingResult(
        intents=intents,
        entities=data.get("entities") if isinstance(data.get("entities"), dict) else {},
        missing_info=_list(data.get("missing_info")),
        need_followup=bool(data.get("need_followup", False)),
        retrieval_targets=retrieval_targets,
        query_type=str(data.get("query_type") or "knowledge_question"),
        user_goal=str(data.get("user_goal") or ""),
        topic_relation=str(data.get("topic_relation") or "same_topic"),
        rewritten_query=str(data.get("rewritten_query") or query),
        satisfied_info=_list(data.get("satisfied_info")),
    )


def analyze_followup_response(
    chat: ChatProvider,
    query: str,
    messages: list[Message],
    state: ConversationState,
) -> FollowupAnalysisResult:
    prompt = f"""
你是劳动法 Agentic RAG 系统的追问回应分析模块。请只输出 JSON，不要输出解释。
任务：分析用户对追问的回应，提取新信息，更新缺失信息，判断是否需要继续追问。
不构成正式法律意见，不承诺胜诉。

## 语言规则
- missing_info（还需要追问的问题）必须使用用户提问的语言：用户用英文提问就用英文追问，用中文就用中文追问。
- entities、satisfied_info 等字段用中文（内部使用）。

## 字段定义

### entities（实体信息，object，只输出用户明确提到的内容，未提及的字段可省略）
注意：被否定词管辖的列举项不应提取为实体。例如"没有书面合同、工资记录"中，书面合同/工资记录均被否定，不要提取。
- subjects: 主体，如 ["用人单位", "劳动者", "员工", "公司"]
- legal_relations: 劳动关系相关，如 ["劳动关系", "劳务关系", "劳务派遣", "非全日制用工"]
- amounts: 数额/标的，如 "5000元"、"三个月工资"、"经济补偿金"
- deadlines: 期限/时间节点，如 "试用期3个月"、"合同期满"、"已过2年"、"仲裁时效"
- jurisdiction: 管辖地/工作地，如 "在深圳工作"、"公司注册地在北京"
- causes: 案由（劳动争议类），如 "拖欠劳动报酬"、"违法解除劳动合同"、"工伤待遇争议"
- crimes: 罪名，如 "拒不支付劳动报酬罪"
- statutes: 劳动法规名，如 "劳动合同法"、"劳动争议调解仲裁法"、"劳动法"、"社会保险法"
- cases: 案号

### missing_info（还需要追问的问题，string[]）与 need_followup（boolean）
missing_info 是用户本次回应后，还需要继续追问的问题，用自然语言、可直接展示给用户的形式写出。

判断规则：
1. 逐一对照下方"已追问的问题"列表，判断用户本次回应是否回应了各项。回应方式包括：肯定确认、否定否认、描述性说明——只要用户提到了即为已回应。
2. 否定即回答：用户说了"没有""无""不是""正常""没""否""没事""不""未"等，表示该情况不存在，已回应，不要再追问。
   特别注意中文列举管辖规则：当"没有""无"等否定词后面跟着用顿号（、）、逗号（，）或"和""与""及""或"等连接的多个项目时，否定词管辖整个列表，列表中所有项目均被否定。例如"没有书面合同、工资记录、考勤记录、转账凭证"表示全部不存在。
   反向规则同等重要：否定词仅管辖其明确列举的项。用户说"没有书面合同"只否定书面合同，不等于否定了工资记录；只有"没有书面合同、工资记录"、"没有书面合同和工资记录"、"没有书面合同等"才表示否定范围覆盖工资记录。不要把未被明确提及的追问项当作已被否定。
   此外注意中文后置否定句式：否定词在关键词之后同样表示否定。如"书面合同倒是没有""合同是没有的""工资记录的话倒没有"均表示该证据不存在。不要因为否定词在关键词后面就误认为该证据被肯定提及。
3. 描述即回答：用户给出描述性语句（如"就是普通欠薪""大概三个月""差不多那样"），即使不够精确也已回应。
4. missing_info 只包含用户**仍未提供**、需要继续追问的问题。已被回应的不纳入。
5. need_followup 仅在存在还需要追问的问题时才设为 true。
6. 用户表示不理解（"不理解""看不懂""什么意思""没听懂""不太明白"）时，将问题用更通俗、与用户提问一致的语言重新表述放入 missing_info，need_followup 设为 true。
7. 去重规则：若 missing_info 中的某条内容与"已追问的问题"列表中的任意一条语义高度重叠，则不要重复纳入。只列出真正新的、未被覆盖的问题。
8. 禁止追问隐私信息：不要追问公司具体名称、个人姓名、身份证号、具体家庭住址、电话号码、银行账号等可识别个人或企业身份的信息。需要了解用人单位情况时，改问泛化信息（如公司类型：国企/私企/外企/个体工商户、所属行业、用工形式等）；需要了解工作地点时，只问城市/地区级别（如"在哪个城市工作"），不问具体地址。

### satisfied_info（用户已确认的关键信息点，string[]）
记录用户已确认的关键信息点，用简洁的自然语言描述，供跨话题信息延续判断使用。例如"当事人身份：劳动者""法律关系：劳动关系""证据：无书面合同"。否定也算已确认。空数组表示没有确认的信息点。

## Few-shot 示例

示例1 —— 否定全部追问，静态信息已满足：
已追问的问题：["是否有书面合同、工资记录等证据？", "当事人身份：劳动者", "法律关系的性质？"]
用户回复：没有书面合同、工资记录那些，都是口头约定。我是劳动者，是劳动关系。
输出：
{{"entities": {{"subjects": ["劳动者"], "legal_relations": ["劳动关系"]}}, "missing_info": [], "need_followup": false, "satisfied_info": ["当事人身份：劳动者", "法律关系：劳动关系"]}}

示例2 —— 部分回答，仍需补充信息：
已追问的问题：["是否已经起诉或申请仲裁？", "有没有书面证据？", "时间节点？"]
用户回复：还没起诉，就是公司去年开始拖欠工资，已经拖了快一年了。
输出：
{{"entities": {{"legal_relations": ["劳动关系"], "deadlines": ["拖欠快一年"]}}, "missing_info": ["具体拖欠金额", "是否仍在职"], "need_followup": true, "satisfied_info": []}}

## 输入

最近上下文：
{json.dumps([to_dict(m) for m in messages[-6:]], ensure_ascii=False)}

当前会话状态：
{json.dumps(to_dict(state), ensure_ascii=False)}

已追问的问题（用户应对这些做出回应）：
{json.dumps(state.asked_questions, ensure_ascii=False)}

上轮未发出的追问（用户可能未回应，可继续追问）：
{json.dumps(state.pending_questions, ensure_ascii=False) if state.pending_questions else "无"}

用户本次回应：{query}
""".strip()
    try:
        raw = chat.chat(
            [
                {"role": "system", "content": "你只输出可解析 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        data = _extract_json(raw)
    except Exception:
        data = {}

    return FollowupAnalysisResult(
        entities=data.get("entities") if isinstance(data.get("entities"), dict) else {},
        missing_info=_list(data.get("missing_info")),
        need_followup=bool(data.get("need_followup", False)),
        satisfied_info=_list(data.get("satisfied_info")),
    )


def merge_followup_into_understanding(
    initial_analysis: dict[str, Any],
    followup: FollowupAnalysisResult,
    state: ConversationState,
    query: str,
) -> QueryUnderstandingResult:
    frozen = initial_analysis
    for item in followup.satisfied_info:
        if item not in state.satisfied_info:
            state.satisfied_info.append(item)
    merged_entities: dict[str, Any] = dict(state.known_entities)
    for key, value in followup.entities.items():
        if not value:
            continue
        current = merged_entities.get(key)
        if isinstance(current, list) and isinstance(value, list):
            merged_entities[key] = sorted({*current, *value})
        else:
            merged_entities[key] = value

    return QueryUnderstandingResult(
        intents=frozen.get("intents") or [],
        entities=merged_entities,
        missing_info=followup.missing_info,
        need_followup=followup.need_followup,
        retrieval_targets=frozen.get("retrieval_targets") or [],
        query_type=frozen.get("query_type") or "legal_consultation",
        user_goal=frozen.get("user_goal") or "",
        topic_relation=frozen.get("topic_relation") or "same_topic",
        rewritten_query=frozen.get("rewritten_query") or query,
    )


def classify_topic_continuation(
    chat: ChatProvider,
    query: str,
    messages: list[Message],
    state: ConversationState,
) -> TopicContinuationResult:
    prompt = f"""你是劳动法 Agentic RAG 系统的话题延续与信息延续判断模块。请只输出 JSON。

任务：用户在上一个问题得到回答后，又发了新消息。请判断两件事：
1. 是否延续话题（is_continuation）：是追问上一个案件，还是开启新话题。
2. 信息延续（retained_entities / retained_satisfied）：无论是否延续话题，判断上一个话题已确认的信息点中，哪些在当前问题里仍然成立、可以沿用。

## 判断标准

**话题延续（is_continuation = true）**：
- 用户对上一轮答案提出质疑、要求补充解释、询问细节
- 用户针对答案中提到的法条/概念进一步提问
- 例："能再详细解释一下吗""这个法条具体怎么理解"

**新话题（is_continuation = false）**：
- 用户提出新的事实、法律关系或问题，与上一轮主题无关
- 例："那我还有加班费没结清的问题"（与上一轮的拖欠工资是不同的争议焦点）

## 信息延续（retained_entities / retained_satisfied）

即使开启新话题，某些信息仍可能延续。请根据当前问题的表述，结合"已知实体"和"已满足信息"，逐项判断哪些仍然成立：

- retained_entities：从"已知实体"中筛选出当前问题里仍然成立、仍可沿用的键值对。已经改变或与当前问题无关的，不要放入。
- retained_satisfied：从"已满足信息"中筛选出当前问题里仍然成立、仍可沿用的项。已经改变或不再适用的，不要放入。

判断要点：
- 身份（当事人是劳动者还是用人单位）是"每个案件各自独立"的：新话题里用户角色可能改变，只有在当前问题明确沿用同一角色时才延续。
- 法律关系（如劳动关系）若两个话题都是同一类纠纷，通常可延续。
- 金额、时间节点、证据等通常是"具体案件"专属，新话题一般不延续（除非当前问题明确指向同一案件）。

## 字段

- is_continuation: boolean
- rewritten_query: string，话题延续时给出检索改写 query；新话题留空。注意：rewritten_query 始终用中文（法条库为中文）。
- retained_entities: object，延续的实体（可为空对象 {{}}）
- retained_satisfied: string[]，延续的已满足信息（可为空数组 []）

## Few-shot 示例

示例1 —— 话题延续：
已知实体：{{"legal_relations": ["劳动关系"], "amounts": ["三个月工资"]}}
已满足信息：["当事人身份：劳动者", "法律关系：劳动关系"]
用户本次输入：能详细说说仲裁时效怎么算吗？
输出：
{{"is_continuation": true, "rewritten_query": "劳动争议 仲裁时效", "retained_entities": {{"legal_relations": ["劳动关系"], "amounts": ["三个月工资"]}}, "retained_satisfied": ["当事人身份：劳动者", "法律关系：劳动关系"]}}

示例2 —— 新话题，身份改变：
已知实体：{{"subjects": ["公司"], "legal_relations": ["劳动关系"], "amounts": ["三个月工资"]}}
已满足信息：["当事人身份：劳动者", "法律关系：劳动关系"]
用户本次输入：我的员工离职后要竞业限制怎么办？
输出：
{{"is_continuation": false, "rewritten_query": "", "retained_entities": {{"legal_relations": ["劳动关系"]}}, "retained_satisfied": ["法律关系：劳动关系"]}}

## 输入

已知实体（上一个话题抽取并累积的）：
{json.dumps(state.known_entities, ensure_ascii=False)}

已满足信息（上一个话题已确认的追问项）：
{json.dumps(state.satisfied_info, ensure_ascii=False)}

最近对话上下文：
{json.dumps([to_dict(m) for m in messages[-6:]], ensure_ascii=False)}

用户上一条消息（系统回复，仅供参考）：
{json.dumps(messages[-2].content if len(messages) >= 2 else "", ensure_ascii=False)}

用户本次输入：{query}
""".strip()
    try:
        raw = chat.chat(
            [
                {"role": "system", "content": "你只输出可解析 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        data = _extract_json(raw)
    except Exception:
        return TopicContinuationResult()

    retained_entities = data.get("retained_entities")
    if not isinstance(retained_entities, dict):
        retained_entities = {}

    return TopicContinuationResult(
        is_continuation=bool(data.get("is_continuation", False)),
        rewritten_query=str(data.get("rewritten_query") or ""),
        retained_entities=retained_entities,
        retained_satisfied=_list(data.get("retained_satisfied")),
    )
