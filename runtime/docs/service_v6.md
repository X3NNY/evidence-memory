# 文本服务 v6：剩余4题审计与预算内原文补全

本轮只用原100题和原缓存，没有模型 API、新 subagent 或新问答/Judge评分。原始 LoCoMo 22/50（44分）、冻结离线 v7 及 v2–v5 归档保留。

## 4题具体缺在哪里

| 问题 | v5实际轮次覆盖@48k | 缺口性质 | 本轮结果 |
|---|---:|---|---|
| John做过哪些武术 | 63.04% | kickboxing、taekwondo都已出现；跆拳道所在短尾块缺发言开头及说话人 | 补齐为100%，恢复原文中的John标签 |
| John除了篮球还喜欢什么运动 | 66.94% | surfing已出现；遗漏一条冗余篮球引用及部分图片说明 | 保留原值，未改金标 |
| Andrew收养Toby前遇到什么困难 | 75.00% | 找房、犬种与空间、公园条件都已出现；未覆盖引用是结尾致谢 | 保留原值，未改金标 |
| Audrey与Andrew可能住在哪个州 | 0% | 金标指向公园地图；缓存文字及图片说明没有州名，全部缓存文字没有Minnesota字面线索 | 当前文字输入不足以核实答案，仍计入分母 |

这些判断来自本地原文审计，不是新的回答或评分。州名题未加载原图，不能断言原图是否支持标准答案；图片URL和图片搜索query均未加入检索输入。完整引用覆盖低不等于缺少同等比例的答案事实。

## 通用补全方法

v5 的完整块排序不变。先计算消费者字符预算内的原始完整记录前缀，再保留这个前缀的全部原文，仅利用余量恢复跨块的同一轮发言。优先修复没有明确说话人标签的短续段，再按额外字符成本补其他边界；最多附加128词、2048字符。所有额外内容来自原文字符区间，不加入事实、同义词、题号规则或人物常量。

输出窗口保留原锚点分数与创建时间；窗口ID由锚点与源区间确定，重启可重建。原始消息和向量缓存不改写。普通角色消息沿用原输出。窗口及重叠记录的同一原文词只计一次，评测器从原始数据重建每个可见词的位置。

`AML_WINDOW_CONTEXT_CHARS=48000` 是补充原文所用的消费者预算，0关闭；它不会截断Search返回序列，原top_k数量和顺序仍保留。部署时需与实际消费者预算一致。增加3轮相邻对话的对照在同100题上没有进一步证据收益，因此默认只补当前跨块发言。

## 同100题结果

| LoCoMo指标 | v5 | v6 |
|---|---:|---:|
| 来源召回@10 | 74.00% | 74.00% |
| 来源召回@48k | 97.00% | 97.00% |
| 实际答案轮次 token 覆盖@48k | 96.10% | 96.84% |
| 全部金标轮次完整@48k | 46/50 | 47/50 |

1题完整性改善、49题实际轮次覆盖持平，无新增退步；LongMemEval全部50题响应逐条一致。新窗口改变部分返回内容及ID，未宣称LoCoMo响应字节不变或产生新的问答成绩。来源召回没有提升，这轮收益是原文完整性与说话人归属。

## 复核与局限

- `runs/text_remaining_four_001/diagnosis.json`：4题逐条引用、原文和源区间覆盖，人工分类仅供评测。
- `runs/text_source_windows_probe_001/`：固定两个补全方案、100题缓存重放和候选选择；未构造embedding provider。
- `runs/text_source_windows_default_002/`：最终默认API逐条复现100题新输出与窗口来源；与已独立复核的001结果一致。
- `runs/text_hybrid_release_006/`：最终测试、报告与部署包。

```powershell
$env:AML_ALLOW_EMBEDDING_API = '0'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest -q
python scripts/verify_windows_release.py --output runs/text_source_windows_default_repeat
python scripts/audit_service_turn_coverage.py --runs runs/text_source_windows_default_repeat
python scripts/smoke_hybrid_service.py --output runs/text_hybrid_local_repeat
```

研究复核需完整本地数据和缓存；部署包不含数据、答案或凭据。默认API验证缺缓存即失败并阻断外网，原embedding缓存哈希不变，原44分对应537份文本结果与报告逐个核对。内核与离线文本适配器源码不变，继承此前已通过的冻结v7回归。

原100题已反复用于开发，并非新题盲测。输入仍为历史256词块及会话分组，官方原始消息格式尚未验证。没有公网部署、Docker镜像构建或官方Smoke/Full。
