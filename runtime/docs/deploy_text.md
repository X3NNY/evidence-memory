# 文本服务部署与提交

版本 text-http-hybrid-v8。存储事务、generation快照、检索和原文证据组装分层，向量缓存每批最多读取200条。与HTTP v7保持同100题的完整响应和来源一致，见 [v8架构与验证](service_v8.md)。保留v7的会话头去重、原文窗口和完整轮次补充策略，详见 [v7检索方法](service_v7.md)。默认hybrid、旧全局正文RRF开关为0、远程模型访问关闭。这是HTTP服务版本，与冻结离线v7分别编号。本轮未部署公网、构建Linux镜像或运行官方Smoke/Full。

## 本地零费用检查

```powershell
python scripts/smoke_hybrid_service.py --output runs/text_hybrid_local_repeat
python -m memory_bench.offline_text_regression --output runs/text_hybrid_v7_repeat
python scripts/verify_frames_release.py --output runs/text_session_frames_default_repeat
python scripts/audit_service_turn_coverage.py --runs runs/text_session_frames_default_repeat
```

第一项自行创建合成向量、凭据、数据库，启动两个独立本机 HTTP 进程，验证后自动关闭；不访问外部模型。100题重放只读原缓存，缺失立即失败。

生产使用独立数据库，例如 var/text-hybrid-v8.sqlite3。v2–v7 的混合向量和原文格式可复用，无需重新嵌入；升级并重启后重建内存索引。旧 BM25 库或不同 embedding namespace 会拒绝混用，避免成功写入却没有对应向量。

## 未来部署时的显式配置

| 变量 | 用途 |
|---|---|
| AML_MEMORY_SYSTEM_KEY_FILE | 我们服务的评测专用鉴权密钥文件 |
| AML_RETRIEVER=hybrid | 固定混合检索 |
| AML_EMBEDDING_CONFIG=configs/text_service_hybrid.json | 无密钥配置 |
| AML_EMBEDDING_MODE=remote | 对官方新数据使用 v4 |
| AML_ALLOW_EMBEDDING_API=1 | 显式开启远程访问；本轮未开启 |
| DASHSCOPE_API_KEY | 模型凭据 |
| DASHSCOPE_EMBEDDING_URL | 已验证的 DashScope 原生 embedding 请求地址，不是 compatible-mode base_url |
| AML_BODY_LEXICAL_WEIGHT=0 | 禁用旧版全局额外正文 RRF；v4 对明确具名对话仍自动进行双粒度融合 |
| AML_QUERY_CACHE_ENTRIES=16 | 每个用户索引最多缓存16条查询；0关闭 |
| AML_QUERY_CACHE_BYTES=4194304 | 每个用户索引查询缓存的保守估计字节上限；0关闭 |
| AML_WINDOW_CONTEXT_CHARS=48000 | 补充原文时使用的消费者字符预算；需与实际消费者一致，0关闭补全 |
| AML_SESSION_FRAMES=1 | 启用具名对话会话合并；0回退 v6 输出视图，原文窗口开关另行保留 |

查询缓存仍在每次请求前核对数据库 generation，新写入后新索引不会复用旧证据记录。缓存保存不可变 Search 证据和计入字节预算的来源元数据；对外只返回原4字段的新字典，失败不缓存。默认8个常驻 LRU 用户索引的查询缓存估计上限为32 MiB；进行中的请求可能暂时持有已淘汰索引，进程还包括原文、向量、Python 与排序开销。并发构建按用户合并，读锁与 Add 的外部 embedding 工作互不阻塞。详见 [v3验证说明](service_v3.md)。

会话合并后，一条 Search 记录可能包含多个原始块，记录数不会超过 top_k。48k预算用于保留并补充上下文，不是 HTTP 返回大小上限。下游仍需执行自己的上下文预算；返回粒度改变后，固定 Top-10 指标不能直接与旧块粒度比较。

平台 Eval Key、我们生成的 Memory System Key、模型 Key 各自独立。服务不需要 Answer/Eval 配置，正式回答与评分由平台执行。

```text
python -m memory_bench.service --host 0.0.0.0 --port 8787 --database var/text-hybrid-v8.sqlite3
```

Dockerfile / compose.yaml 已提供。Compose 使用持久化命名卷、独立系统密钥文件，默认映射服务器127.0.0.1:8787，需要配置公网 HTTPS 反向代理。cache_only 不能处理官方未见数据，必须完成上述显式配置。部署时记录镜像 digest 和 /app/installed-packages.txt；当前仅固定已验证的直接依赖，不声称镜像已构建或所有传递依赖已锁定。

## 上线步骤

1. 从服务器外验证 HTTPS Health、鉴权、Add→Search、幂等重试、重启与数据卷。
2. 少量合成新数据完成真实 v4 联调，确认限流、耗时、费用和实测容量。配置为单进程并发16、embedding 批量10、30秒超时、最多2次重试，不等于 Full 容量已验证。
3. 公开仓库并固定 commit，补齐团队、联系人、来源和方法改动。当前目录不是 Git 仓库，公开地址和 commit 待确定。
4. 提交完整准入申请，绑定正式接口与版本，拿到 Eval Key 后运行官方 Smoke。
5. 稳定后启动 Full，保持服务和数据卷可用，按官方要求留存和删除记忆；日志不记录原文、查询或凭据。

[API Guide](https://agentmemoryleaderboard.ai/api-guide)，[比赛时间](https://agentmemoryleaderboard.ai/competition/)：材料与申请2026-10-31 23:59截止，评测2026-11-04 23:59停止（UTC+8）；部署前核对平台当时规则。
