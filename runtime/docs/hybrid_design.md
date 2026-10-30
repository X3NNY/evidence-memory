# 文本服务 v2：设计依据与验证

原始 LoCoMo 22/50（44分）保持不变；本轮只验证检索和服务，没有生成新答案。

## 项目参考

| 一手来源 | 设计 | 本次采用 |
|---|---|---|
| [Graphiti 搜索配置](https://github.com/getzep/graphiti/blob/main/graphiti_core/search/search_config_recipes.py) | BM25、cosine 多路召回，RRF/MMR 排序 | 复用已有 MemoryEngine、HybridCandidateRanker 和 v7 参数：BM25 0.3、RRF k=20、MMR lambda=0.7 |
| [Graphiti 搜索执行](https://github.com/getzep/graphiti/blob/main/graphiti_core/search/search.py) | 先限定 group，再召回 | 原文、向量、查询缓存、索引均按完全相同的 user_id 隔离 |
| [LlamaIndex 句子窗口](https://docs.llamaindex.ai/en/v0.10.34/examples/node_postprocessor/MetadataReplacementDemo/) | 细粒度检索，返回包含上下文的原文 | 128词段落排序后返回最多256词的完整原文块 |

以上为设计参考，没有复制外部仓库代码。图谱抽取、LLM 摘要、查询改写和 cross-encoder 尚无本轮验证结果，本轮采用无需新增模型的组合。

## 与离线 v7 的对应

共享 BM25、dense cosine、RRF/MMR、PassageQuery、ContentQuery 及其角色、长对话、会话策略。OnlineConversationIndex 用稳定原文偏移替代 benchmark 的固定 chunk ID。HTTP 按 top_k 返回完整记录，不施加原本地48,000字符证据预算，由平台控制 Answer 上下文。因此这是 v7 策略的线上适配，不能直接引用离线 v7 的96.67%/98.40% token 覆盖率作为服务分数。

Add 只切分新增消息，不移动旧边界、不重扫全部原文。每块最多256词、有效正文6000 UTF-8字节，支持无空格中文和超长代码，完整原文与偏移保留。embedding 只编码正文；输出保留实际角色及已有来源时间，不推断事件时间。

文档和查询向量分别以 user_id、模型/端点/dimension namespace、类型、正文哈希持久化。重试、相同查询和重启均可复用，跨用户不共用向量。长查询分块池化，完整查询仍参与词法检索。只在 embedding 完成且原始消息持久提交后确认 Add，失败503可重试，已成功的向量工作供重试复用。Search 不重新嵌入文档；generation 使其他进程的旧 LRU 索引失效。

完全相同的返回证据去重，角色或来源时间不同的记录保留；原始写入不做语义删除。默认 cache_only，缺缓存503；只有 remote 与 AML_ALLOW_EMBEDDING_API=1 同时显式配置才访问 text-embedding-v4，没有静默 BM25 降级。

## 固定100题

`runs/text_hybrid_cache_001` 经实际 Add/Search ASGI 路由重放100题、60个用户范围，只读历史向量。零费用输入采用原已缓存的256词块及原会话分组；这不是官方原始写入质量测试，也不是新题盲测。金标、答案和来源标签只在评测器中使用。

| 指标 | 数据集 | 旧 HTTP BM25 | 默认混合 | 变化（百分点） |
|---|---|---:|---:|---:|
| 前10条来源召回 | LoCoMo | 40.50% | 69.17% | +28.67 |
| 前10条来源召回 | LongMemEval-S | 41.74% | 76.21% | +34.47 |
| 同为48,000字符完整记录前缀的来源召回 | LoCoMo | 69.00% | 93.00% | +24.00 |
| 同为48,000字符完整记录前缀的来源召回 | LongMemEval-S | 66.21% | 93.01% | +26.81 |

来源召回只表示返回块是否涉及标注来源，不证明完整答案片段已出现，不是 QA 分数。LongMemEval 的50题中3题为 abstention，来源召回分母47。

固定0.25的额外正文词干 RRF 候选将前10条来源召回改为 LoCoMo 70.50%、LongMemEval 72.13%；后者比默认混合低4.08个百分点，因此默认权重为0。候选可设 AML_BODY_LEXICAL_WEIGHT=0.25 重现。原100题的冻结 v7 回归另行重算，6份产物逐字节一致，原始44分及原生 Judge、审计均保留。
