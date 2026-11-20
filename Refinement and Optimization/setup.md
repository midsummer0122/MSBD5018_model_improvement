# Agentic RAG 劳动法问答系统：团队复现与测试说明

## 这个文档是一开始写的，里面提到的有些代码或者文件可能已经删除了，表述上可能也可能有些前后矛盾的地方，但不影响整体阅读。
## 该文档仅供参考，如果哪里流程不懂可以看看。

更新日期：2026-11-21。本文对应当前改进后的项目，原项目根目录为 `G:\MSBD5018_agentic-rag-law-main`。

队友可以把项目解压到自己的目录。**下文所有命令都在项目根目录执行，也就是能看到 `app.py`、`rag`、`scripts` 的位置，不是在本文所在的 `Refinement and Optimization` 中执行。**

这个项目面向劳动法问答：先理解问题，在事实不足时追问，再检索法条和案例，读取完整上下文，生成带来源的回答，最后做安全审查。回答仅供学习与参考，不构成正式法律意见。

## 1. 这个版本做了哪些改进

| 改进 | 当前实现 | 使用时需要知道什么 |
| --- | --- | --- |
| 补充劳动案例库 | 将174份案例PDF处理成父子块，生成向量，接入独立案例集合 | 原有法条数据保留；法条、案例分别建库 |
| 双库检索与统一重排 | 分别召回法条、案例，再统一计算相关性分数 | 两类资料都能参与最终证据选择 |
| 按来源分配名额 | 在子块重排选择和父块证据选择两个阶段，按意图安排法条／案例数量 | 数量不足时可以由另一来源补齐；不是每题强制固定比例 |
| MySQL父块读取 | `documents`、`parent_chunks` 保存文档和父块，运行时优先查询MySQL | MySQL连接失败或缺少父块时，保留本地JSON回退 |
| 可选混合检索 | Dense、Dense＋BM25、Dense＋BM25＋Exact三种模式，可从网页选择 | **默认Dense；不会根据问题自动切换为Hybrid** |
| 轻量证据缺口检查 | 针对法律咨询做一次诊断，必要时单轮补检、统一重排，最多替换一条证据 | **默认关闭，需要时手动开启，会增加模型调用和等待时间** |
| 前端过程展示 | 右侧显示各检索通道、父块回补、重排与来源配额；开启覆盖检查后还显示诊断、补检词和比较分数 | 展示实际执行状态；没有启用的通道明确显示未启用 |

默认流程：

```text
用户问题 → 问题理解／必要时追问 → 法条与案例分别Dense召回
       → MySQL优先回补父块（缺少时读本地JSON）
       → 统一重排与来源配额 → 最多5条父块证据
       → 回答生成 → 安全审查 → 答案与引用
```

选择Hybrid时，在首轮召回中增加BM25和Exact，并在每个库内用RRF融合，再接后面的父块回补和统一重排。证据覆盖检查是**另一个独立选项**，位于首轮证据构建之后、回答生成之前。

### 当前知识库规模

以下数量已按本地JSON和向量CSV重新核验，不以文件名或ID前缀猜测来源。

| 类型 | 文档 | 父块 | 子块／已有向量 | 默认Qdrant集合 |
| --- | ---: | ---: | ---: | --- |
| 法条 | 128 | 511 | 1,384 | `law_statutes` |
| 案例 | 174 | 349 | 2,334 | `law_cases` |
| 合计 | 302 | 860 | 3,718 | 两个集合分别保存 |

向量为 `text-embedding-v4` 生成的1024维向量，集合使用Cosine距离。法条复用了已有切分与向量，案例库是本次新增处理的部分。

### 为什么不默认打开全部优化

在已有20题诊断集上，Dense＋来源配额的综合父块Recall@5为 **0.8083**；两种混合检索配置均为 **0.7667**。覆盖检查开启后综合Recall仍为 **0.8083**，综合MRR由 **0.9250降为0.9000**，同时增加了调用成本。因此当前保留这些能力供按需测试，默认使用Dense，覆盖检查关闭。

这是小规模探索性诊断的结果，不代表所有问题都适用，也不是最终法律回答质量的证明。详细过程见同目录 [report1.md](report1.md)、[report2.md](report2.md)、[report3.md](report3.md)。

## 2. 队友拿到压缩包后，先确认什么

**源码包不等于已经配置好的云服务和数据库。** 当前子块保存在发起人新建的云端Qdrant，父块MySQL在发起人的本机；压缩源码不会把这两个服务自动迁移给队友。

运行真实问答需要：

1. 自己可用的DeepSeek、阿里云Embedding与Rerank访问配置。
2. 能读取两个集合的Qdrant配置：使用团队分配的访问权限，或者把已有向量导入自己的集群。
3. 两个本地JSON目录。它们支持父块回退、BM25／Exact检索和来源识别，即使安装MySQL也建议完整保留。

可以选择两条路线：

| 路线 | 需要准备的内容 | 适合的场景 |
| --- | --- | --- |
| A：先把网页跑起来 | 真实模型接口＋有数据的Qdrant＋本地JSON；保留mysql配置段 | 先体验问答；父块允许从JSON回退 |
| B：完整复现数据库接入 | A的全部内容＋本机MySQL、兼容表结构、860个父块 | 验证生产流程确实通过MySQL回补父块 |

两条路线都不是完全离线方案。JSON可以替代父块数据库读取，但不能替代当前Dense流程中的真实Embedding、Qdrant与模型服务。

## 3. 解压与安装依赖

本项目已在Windows、PowerShell、Python **3.10.6** 环境运行验证。建议队友先使用Python 3.10复现；其他版本需要在自己的机器上验证兼容性。

```powershell
# 改成你解压后的项目根目录
Set-Location "G:\MSBD5018_agentic-rag-law-main"

# 创建属于本机的虚拟环境，不直接复用别人压缩包里的.venv
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

后面直接调用虚拟环境中的Python，因此不需要先执行Activate，也不需要为激活脚本修改PowerShell执行策略。

主要依赖是Flask、PyMySQL、PyYAML、PyMuPDF和jieba等，均在根目录 `requirements.txt` 中。Playwright只用于界面截图测试，不是启动网页的必要依赖。

## 4. 创建自己的配置文件

程序默认读取项目根目录下的 `api_key/api_config.yaml`。**当前包内不应包含发起人的私人配置；本地目录也没有原README提到的 `api_config - template.yaml`，不要直接照搬原来的复制命令。**

先创建目录和文件：

```powershell
New-Item -ItemType Directory -Path api_key -Force | Out-Null
notepad api_key\api_config.yaml
```

将下面的示例保存为UTF-8，并在本机编辑器中替换占位符。示例不包含真实密钥、私人Qdrant地址或阿里云工作空间域名。

```yaml
llm:
  provider: deepseek
  base_url: "https://api.deepseek.com"
  api_key: "填写你自己的DeepSeek密钥"
  chat_model: "deepseek-chat"
  reasoning_model: "deepseek-reasoner"
  timeout_seconds: 60

embedding:
  provider: dashscope
  base_url: "https://<你自己的阿里云服务域名>/compatible-mode/v1"
  api_key: "填写你自己的Embedding密钥"
  model: "text-embedding-v4"
  dimension: 1024
  timeout_seconds: 60

rerank:
  provider: dashscope
  base_url: "https://<你自己的阿里云服务域名>/compatible-api/v1/reranks"
  api_key: "填写你自己的Rerank密钥"
  model: "qwen3-rerank"
  timeout_seconds: 60

qdrant:
  provider: qdrant
  url: "https://<你自己的Qdrant集群地址>"
  api_key: "填写该集群的访问密钥"
  collection_name: "law_statutes"
  cases_collection_name: "law_cases"
  timeout_seconds: 30

mysql:
  host: "127.0.0.1"
  port: 3306
  user: "law_rag_app"
  password: ""
  database: "law_rag"
  charset: "utf8mb4"
```

配置时注意调用代码的实际拼接规则：

| 配置字段 | 代码如何使用 | 填写要点 |
| --- | --- | --- |
| `llm.base_url` | 追加 `/chat/completions` | 不把完整chat路径重复填入基础地址 |
| `llm.chat_model` | 作为聊天模型名 | 字段名是 `chat_model`，不是 `model` |
| `embedding.base_url` | 追加 `/embeddings` | 使用自己账户可用的兼容接口基础地址 |
| `embedding.dimension` | 请求体中转为 `dimensions` | 配置字段用单数 `dimension`；保持1024，与已有向量一致 |
| `rerank.base_url` | 直接请求该URL | 填完整重排地址，代码不会自动补 `/reranks` |
| `qdrant.url` | 追加collection路径 | 填集群根地址，不包含某个集合的路径 |
| `mysql` | 运行时连接父块库 | 路线A也保留该段；路线B填写自己实际账号与密码 |

阿里云示例中的两个地址仅说明本项目使用的路径形态。**具体服务域名、区域和可用模型以队友自己控制台的配置为准**，不能复制示例中的尖括号地址后直接运行。

### 检查是否误用了Mock

```powershell
.\.venv\Scripts\python.exe -X utf8 -c "from rag.providers import Providers; p=Providers(); print({n: type(getattr(p,n)).__name__ for n in ('chat','embedding','rerank','qdrant')})"
```

预期类名为 `ChatProvider`、`EmbeddingProvider`、`RerankProvider`、`QdrantClient`。如果出现 `Mock...`，说明对应密钥缺失，不能把占位回答当作真实问答成功。这条命令只检查服务封装类型，不发请求，也不能证明密钥有效或网络连通；还需要后面的实际网页测试。

## 5. 准备Qdrant：优先复用已有向量

### 已能访问团队的两个集合

填写分配给自己的地址和访问配置，确认集合已经有数据后，可以直接跳到第6或第7节。**不需要重复切分、生成向量或上传。**

### 需要导入自己的新集群

已有向量在以下目录，无需再次购买批量Embedding调用：

- 法条JSON：`knowledge_data/json`；向量CSV：`knowledge_data/embedding_csv`。
- 案例JSON：`knowledge_data/json_cases`；向量CSV：`knowledge_data/embedding_csv_cases`。

先做本地dry-run，不连接Qdrant：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\upload_child_embeddings_to_qdrant.py --json-dir knowledge_data\json --csv-dir knowledge_data\embedding_csv --collection law_statutes --dry-run
.\.venv\Scripts\python.exe -X utf8 scripts\upload_child_embeddings_to_qdrant.py --json-dir knowledge_data\json_cases --csv-dir knowledge_data\embedding_csv_cases --collection law_cases --dry-run
```

当前数据的预期输出分别是 `points ready: 1384` 和 `points ready: 2334`，维度1024，距离Cosine。本文编写时已实际执行两条命令验证通过。

确认目标是自己的集合后，再执行上传：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\upload_child_embeddings_to_qdrant.py --json-dir knowledge_data\json --csv-dir knowledge_data\embedding_csv --collection law_statutes
.\.venv\Scripts\python.exe -X utf8 scripts\upload_child_embeddings_to_qdrant.py --json-dir knowledge_data\json_cases --csv-dir knowledge_data\embedding_csv_cases --collection law_cases
```

脚本读取YAML中的Qdrant配置，默认按chunk_id生成稳定UUIDv5作为point ID；目标集合不存在时创建，存在时执行upsert。**不要加 `--recreate`，它会删除并重建集合。** upsert虽然不删除整个集合，但会更新相同point ID，使用已有集合前应核对内容和向量配置。

两条命令都显式指定了 `--collection`，避免案例也被导入默认法条库。集合名称若自行修改，命令参数与YAML必须一致。文件名和关联ID应保持字符串，不能用Excel把前导零转成数字。

只有新增资料、现有向量缺失或模型不兼容时，才需要另行研究重新切分／向量化。相关脚本是 `scripts/chunk_case_pdfs.py`、`scripts/embed_json_children_ali.py`；它们不是拿到当前包后的必跑步骤，批量向量化会产生额外费用。

## 6. 父块读取：JSON起步或MySQL完整复现

### 路线A：暂不安装MySQL

保留YAML中的mysql配置段和两个JSON目录。运行时会先尝试MySQL，连接不可用时从本地JSON取回缺少的父块，因此可以先体验双库问答。

网页能回答问题只能说明流程可用，**不能据此证明MySQL连接成功**。如果选择这条路线，应把测试结果标记为“本地JSON父块回退”。Qdrant仍须准备好，不能因为有JSON就跳过向量库。

### 路线B：配置自己的MySQL

本项目原环境使用MySQL 8.4，本地数据库名为 `law_rag`；队友应安装、启动自己的实例，不复制发起人的MySQL程序目录、数据目录或Windows服务配置。

1. 在MySQL Workbench或管理员SQL会话中，打开并执行项目的 `scripts/mysql_schema.sql`。已有表先核对结构。该文件使用utf8mb4，保留字符串ID，父块正文为LONGTEXT，建表使用IF NOT EXISTS，不会自动升级已有的不兼容表。
2. 使用自己的本机项目账号。可以在Workbench中创建 `law_rag_app`，设置密码，并授予其 `law_rag` 库的SELECT、INSERT、UPDATE权限；建库建表由管理员执行。无需开放公网，也不要给项目账号全局管理员权限。
3. 在本机YAML中填写实际连接信息，保留mysql段。不要把密码写入README、聊天、命令参数或截图。

先检查两类父块，dry-run不连接MySQL：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\import_parent_chunks_to_mysql.py --json-dir knowledge_data\json --dry-run
.\.venv\Scripts\python.exe -X utf8 scripts\import_parent_chunks_to_mysql.py --json-dir knowledge_data\json_cases --dry-run
```

当前预期为128份文档／511父块、174份文档／349父块，本文编写时已执行验证通过。

正式导入时，在本地隐藏输入密码：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\import_parent_chunks_to_mysql.py --json-dir knowledge_data\json --host 127.0.0.1 --port 3306 --user law_rag_app --database law_rag --ask-password
.\.venv\Scripts\python.exe -X utf8 scripts\import_parent_chunks_to_mysql.py --json-dir knowledge_data\json_cases --host 127.0.0.1 --port 3306 --user law_rag_app --database law_rag --ask-password
```

**导入脚本使用命令行参数／MYSQL环境变量，不自动使用YAML中的MySQL配置。** 上面的主机、端口、账号若与自己的实例不同，应一并修改；YAML另行配置供运行时读取。脚本检查重复ID、父子关联和同ID内容冲突，相同内容可以幂等导入，冲突需要先查明。

导入后进行不允许JSON回退的完整验证：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\verify_mysql_parent_store.py
```

该命令使用YAML连接，核对本地全部预期父块、正文、标题、文档关联及哈希，并让本次验证中的JSON回退直接失败。无额外资料的干净数据库应有302文档、860父块；成功记录包括 `runtime_mysql_parents: 860`、`json_fallback_calls: 0`。

此验证脚本还检查 `law_rag`、utf8mb4及项目账号权限，建议使用上面约定的数据库和项目账号。这里只运行默认核验，不需要添加会执行真实检索与Rerank的 `--retrieval`。

## 7. 启动网页、选择模式和停止服务

```powershell
.\.venv\Scripts\python.exe -X utf8 app.py
```

浏览器打开 **http://127.0.0.1:5000**。终端显示Flask开发服务器提示、`Running on http://127.0.0.1:5000` 后保持等待，是正常运行状态。这个终端要保持打开，停止项目按 **Ctrl+C**。

输入框上方有两个独立设置：

| 设置 | 默认值 | 如何测试 |
| --- | --- | --- |
| 检索模式 | Dense语义检索 | 下拉可切到Dense＋BM25或混合检索Dense＋BM25＋Exact；对下一次提交生效 |
| 证据覆盖检查（可选） | 关闭 | 勾选后再提问；仍仅对满足门禁条件的法律咨询执行诊断 |

刷新页面或新建会话，恢复Dense和覆盖检查关闭；发送过程中选择控件会锁定。切换下拉选项不会把上一轮已完成的路径重新标成另一模式，右侧显示的是该轮实际执行的检索。

**系统会自动判断意图、是否追问和来源配额，但不会自动在Dense与Hybrid之间择优切换。** 覆盖检查即使发生替换，也会提示“gap not rechecked”，不能据此声称缺口已经解决。

修改Python代码后需重启服务；前端更新后按Ctrl+F5强制刷新。MySQL是独立服务，关闭Flask不会停止MySQL。需要启动MySQL时，可在Windows服务管理器中启动自己的实际服务名；原机器的 `MySQL84LawRag` 名称不保证在队友机器上存在。

## 8. 建议按这个顺序做少量体验测试

### 测试一：信息不足时是否追问

> 公司要辞退我，我能拿到什么补偿？

观察系统是否进一步询问必要事实。具体追问内容可能变化，不要求和截图逐字相同。

### 测试二：默认Dense的双库问答

保持覆盖检查关闭：

> 我是上海劳动者，在职2年、月薪10000元，公司拖欠了最近两个月工资。我有劳动合同、考勤记录和银行流水，仍在职，尚未投诉或申请仲裁。请说明追索欠薪的法律依据和可行步骤，并结合相关案例解释。

观察右侧是否出现两个库的召回、父块回补、统一重排与证据数量；回答末尾是否有与证据来源对应的引用。信息已较完整也不保证模型绝不追问，按实际追问继续补充即可。

### 测试三：手动切换Hybrid

选择“混合检索：Dense＋BM25＋Exact”，覆盖检查仍关闭，新建问题时注意重新选择模式：

> 请查找指导案例190号王山与万得信息案，说明仅营业执照经营范围相同为什么不足以认定竞业竞争关系，并联系劳动合同法第二十四条；不涉及本人纠纷。

看右侧Dense／BM25／Exact是否启用、是否显示库内RRF，再接父块回补和统一重排。默认Dense下BM25、Exact和RRF显示未启用是正常表现。Exact不是出现任意数字就必然命中，要以实际通道结果为准。

### 测试四：按需打开覆盖检查

仅想体验这部分时再勾选：

> 我是上海劳动者，女，劳动关系存续2年、月薪10000元，2026年10月处于怀孕5个月，公司因客观情况变化拟解除且没有违纪事由。我有劳动合同、孕检和拟解除通知，尚未仲裁。公司能否适用普通无过失解除，合同年底到期时又该怎样处理？

该题曾在诊断实验中触发一次补检并替换一条证据，但实时运行结果可能不同。卡片会区分免检、判充分、诊断失败、补检未替换或已替换；不要把“没有替换”直接认作功能异常。

真实网页测试会产生模型费用，先做以上少量测试即可。关闭覆盖检查只省掉覆盖阶段的额外调用，原有问答仍正常调用模型。

## 9. 代码位置与已有实验资料

以下路径均相对项目根目录。复现时使用根目录下的实际代码，不要用报告目录中的旧副本覆盖它们。

| 文件／目录 | 作用 |
| --- | --- |
| [app.py](../app.py) | Flask入口、普通／流式API，接收检索模式和覆盖开关 |
| [rag/workflow.py](../rag/workflow.py) | 追问、检索、覆盖检查、回答和安全审查的编排 |
| [rag/skill_router.py](../rag/skill_router.py)、[rag/source_policy.py](../rag/source_policy.py) | 意图映射、默认模式、来源识别与两阶段配额 |
| [rag/retrieval.py](../rag/retrieval.py)、[rag/lexical.py](../rag/lexical.py) | 双库Dense召回、父块回补、重排，以及BM25／Exact／RRF |
| [rag/evidence.py](../rag/evidence.py)、[rag/answer.py](../rag/answer.py)、[rag/coverage.py](../rag/coverage.py) | 证据聚合、共享证据展示、轻量诊断与单轮受限补检 |
| [rag/schemas.py](../rag/schemas.py) | 检索计划及父子块、证据结构；覆盖开关默认False |
| [static/app.js](../static/app.js)、[static/style.css](../static/style.css)、[templates/index.html](../templates/index.html) | 网页选项、实际步骤与数量显示 |
| [scripts/chunk_case_pdfs.py](../scripts/chunk_case_pdfs.py)、[scripts/embed_json_children_ali.py](../scripts/embed_json_children_ali.py) | 新资料案例切分与向量生成，已有数据测试无需重跑 |
| [scripts/upload_child_embeddings_to_qdrant.py](../scripts/upload_child_embeddings_to_qdrant.py) | 将现有子块和向量导入指定集合 |
| [scripts/mysql_schema.sql](../scripts/mysql_schema.sql)、[scripts/import_parent_chunks_to_mysql.py](../scripts/import_parent_chunks_to_mysql.py)、[scripts/verify_mysql_parent_store.py](../scripts/verify_mysql_parent_store.py) | 建表、父块导入、MySQL真实读取验证 |
| [eval/queries_ablation_20.jsonl](../eval/queries_ablation_20.jsonl) | 固定20题诊断集 |
| `eval/results_ablation_20`、`eval/results_hybrid_20`、`eval/results_coverage_20` | 原始实验结果、请求缓存、逐题记录和输入核验信息 |
| [eval/test_retrieval_mode.py](../eval/test_retrieval_mode.py)、[eval/test_coverage_light.py](../eval/test_coverage_light.py) | 可选模式、API参数及覆盖边界的离线测试 |
| [eval/verify_retrieval_modes_ui.py](../eval/verify_retrieval_modes_ui.py)、[eval/verify_coverage_ui.py](../eval/verify_coverage_ui.py) | 用已保存真实响应验证网页接线；需要本机MySQL、Playwright及Edge |

离线单元测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s eval -p "test_*.py"
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s scripts -p test_mysql_setup.py
```

当前代码此前已通过47项测试。离线测试中的模拟输入用于验证边界，不代表网页问答使用Mock。两种界面验证脚本使用已保存真实接口响应和本机MySQL，不重新生成法律答案，截图也不能替代一次新的端到端答案质量测试。

不必为了体验网页重跑四组消融、M1—M3或C0—C1。历史结果绑定了题目、模型参数、配置或代码哈希；代码、路径和凭据改变后，旧缓存不一定能在队友机器上直接重放。若要做新实验，应保留原结果，先读对应脚本和报告，再另建结果目录并核算调用费用。

报告阅读顺序：

1. [案例库构建与双库接入优化记录.md](案例库构建与双库接入优化记录.md)：案例资料如何变成可检索知识库。
2. [report1.md](report1.md)：双库统一重排与来源配额。
3. [report2.md](report2.md)：混合检索效果与局限。
4. [report3.md](report3.md)：轻量覆盖检查、实验结果及默认关闭的取舍。

## 10. 打包哪些内容，排除哪些内容

| 内容 | 建议 | 原因 |
| --- | --- | --- |
| `app.py`、`requirements.txt`、`rag/`、`scripts/`、`static/`、`templates/` | 必须保留 | 完整运行代码、依赖及建库工具 |
| `knowledge_data/json/`、`knowledge_data/json_cases/` | 必须保留 | 本地父块回退、来源映射及词法检索需要 |
| `knowledge_data/embedding_csv/`、`knowledge_data/embedding_csv_cases/` | 建议一起提供 | 队友可低成本导入自己的Qdrant，不必重新生成向量 |
| `eval/`、`docs/`、`Refinement and Optimization/` | 团队复现建议保留 | 诊断题集、结果、脚本和说明；本README也在其中 |
| `knowledge_data/劳动案例/`、`knowledge_data/中央法规/` | 有空间时保留 | 用于追溯原文或重新切分；仅用现有数据问答不是必需 |
| `data/` | 排除 | 本机运行记录、日志、会话或临时验证产物，运行时可重新生成 |
| `.venv/`、`venv/`、各级 `__pycache__/`、`*.pyc`、IDE缓存 | 排除 | 不能替代队友自己的环境，且体积大 |
| `api_key/` 整个目录 | **必须排除** | 不仅YAML，TXT、CSV也可能存放真实密钥；队友按第4节自行创建 |
| `.env`、其他账号导出、连接密码、私钥、访问令牌文件 | **必须排除** | 私人凭据不能进入共享包 |

本目录如果还包含以前手动备份的 `rag/`、`scripts/`、`eval/` 等子目录，它们不是当前程序的加载路径，可能早于最新界面修改；可以不随包提供。**以项目根目录下的同名目录为准。** 不需要把MySQL程序、MySQL数据目录或发起人的云集群凭据打包。

`.gitignore`不会自动替压缩软件过滤文件。打包后请检查压缩包内部，尤其确认没有整个 `api_key/`、私人配置备份和日志。本文配置示例可以分享，发起人的真实YAML不能分享。

## 11. 常见问题

| 现象 | 先检查什么 |
| --- | --- |
| 出现Flask开发服务器警告，终端不继续输出 | 属于本地演示正常状态；保持终端开启，再打开浏览器 |
| 网页没有新的下拉框或覆盖开关 | 确认使用顶层最新代码；重启Flask后Ctrl+F5 |
| 默认界面没有运行BM25／Exact | 默认Dense；需要自己选择混合检索，不是系统自动择优 |
| 回答像占位提示或检索为空 | 按第4节检查Mock类型，再检查真实接口配置和两个集合是否有数据 |
| HTTP 401／403 | 在本机确认对应密钥、服务权限、区域与地址；不把密钥或完整请求头贴入群聊 |
| 重排失败后网页仍能出结果 | 原流程可能回退排序；能出答案不等于真实Rerank成功，应看实际状态和调试记录 |
| MySQL没装网页也能回答 | JSON回退正常；若要证明MySQL已生效，运行禁止回退的完整核验 |
| MySQL导入脚本连接的账号不对 | 它不自动读YAML，检查命令参数或MYSQL环境变量 |
| 更换Embedding模型后检索不正常 | 已有向量绑定原模型与1024维；仅改查询模型或维度不能保证兼容 |
| 5000端口被占用 | 先确认是否已有本项目在运行，关闭自己的旧服务；不要结束无关程序 |

本README编写过程中仅做源码与配置字段检查、本地数据统计以及四次dry-run；没有重新上传Qdrant、改写MySQL、生成向量或调用模型。队友机器上的密钥权限、网络和数据库仍需按上述步骤实际验证。
