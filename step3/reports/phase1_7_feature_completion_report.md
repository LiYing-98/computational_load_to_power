# ML.ENERGY Phase 1.7：完整事前特征补齐与建模输入表定稿

## 结论摘要

Phase 1.7 已在任务书边界内完成：针对 ML.ENERGY Benchmark V3 的 **694 条 text-LLM inference run**（其中 **565 条稳定 run**），形成了一行一个 run、字段角色明确、可复核的 GPU 侧建模输入表。本阶段没有训练任何正式预测模型。

核心判断如下：

1. **GPU 侧基础建模可进入下一阶段。** 694/694 run 均具有基础事前配置和五个 GPU 侧 Y 字段；稳定主分析集合为 565 run。
2. **完整 workload 建模应使用单独 cohort。** benchmark token 特征可重构 562/694，effective token 特征可解析 492/694；再排除 28 条无法证实与历史请求向量等价的 GPQA run 后，`eligible_full_workload` 为 **464/694（66.86%）**，稳定子集为 **381/565（67.43%）**。
3. **physics-enhanced 建模只能作为更小的敏感性分析。** physics proxy 原始可用 406/694；再排除其中 23 条未证实历史等价性的 run 后，可靠集合为 **383/694（55.19%）**，稳定子集为 **305/565（53.98%）**。Nemotron hybrid 与 DeepSeek MLA 没有被强行套入不适用公式。
4. **不存在已发现的 X 泄漏。** 92 个允许作为 X 的字段均为执行前已知或可由冻结输入/配置确定计算；实际输出、吞吐、延迟、实际 batch、运行时长、telemetry、稳定性结果以及 GPU Y 均被角色清单排除在 X 之外。
5. **Meta Llama 缺口被显式保留。** 132 条 Llama run 全部保留在 master table，并可参加基础 ex-ante 分析，但因授权访问被拒，不能进入完整 workload 或 physics cohort；没有使用镜像、替代 tokenizer 或替代 config。

## 范围与口径

- 数据范围：`gpqa` 247、`lm-arena-chat` 388、`sourcegraph-fim` 59，共 694 run。
- 稳定性范围：GPQA 187、LMArena 328、Sourcegraph FIM 50，共 565 run。
- 能源口径：ML.ENERGY 结果中的 **GPU 侧 steady-state 能耗/功率及其派生标签**。不把 CPU、DRAM、Node Residual 或整机交流侧能耗混入目标。
- 事前边界：X 只包含任务提交前已知的任务、静态模型、GPU、部署、协议字段，以及由冻结请求、tokenizer、chat template 和静态配置确定计算的 workload/physics proxy。
- 明确排除：Training、Diffusion、CPU/DRAM 建模、Node Residual、全量 raw timeline、Prometheus、模型权重和正式模型训练。

## 主表与角色边界

`master_feature_table` 有 **694 行、152 列、694 个唯一 run key**。角色清单覆盖全部 152 列；92 个 X 字段还必须逐列出现在冻结的 `exante_feature_contract.json` 中，前缀相似但未经审查的新字段不会自动获准进入 X：

| 角色 | 列数 | 是否允许进入 X |
|---|---:|---|
| X_workload（含 physics workload proxy） | 40 | 是 |
| X_model | 41 | 是 |
| X_hardware | 2 | 是 |
| X_deployment | 5 | 是 |
| X_protocol | 4 | 是 |
| Y_gpu | 5 | 否 |
| result diagnostics | 16 | 否 |
| quality diagnostics | 4 | 否 |
| model/split diagnostics | 3 | 否 |
| provenance | 26 | 否 |
| eligibility/restriction | 5 | 否 |
| key | 1 | 否 |

其中 `model_id`/nickname 只作诊断与分组切分，不是默认主模型输入；`is_stable` 只定义事后样本集合，不进入 X。

## 三层资格集合

| 集合 | 全部 run | 稳定 run | 用途 |
|---|---:|---:|---|
| base ex-ante | 694/694（100%） | 565/565（100%） | 下一阶段 GPU baseline |
| full workload | 464/694（66.86%） | 381/565（67.43%） | benchmark + effective workload sensitivity |
| physics feature | 383/694（55.19%） | 305/565（53.98%） | 受支持架构的 physics-enhanced sensitivity |

按任务的 full/physics coverage 分别为：GPQA 149/111，LMArena 256/213，Sourcegraph FIM 59/59。按 GPU 为：B200 235/187，H100 229/196。缺失同时受模型访问、模板实现、历史请求等价性和架构可支持性影响，不能按随机缺失处理。

## Planned workload 审计

- benchmark/client prompt-length 特征可用 562/694；缺失的 132 条全部来自 Meta Llama 授权拒绝。
- 562 个可重构且请求完整的 run 中，523 个与结果 `total_input_tokens` 精确相等，39 个不等，精确率 **93.06%**。
- 39 个差异全部被逐 run 枚举，集中在 GPQA/H100 的重复配置组合。固定源码表明 `num_request_repeats > 1` 时应复制同一请求向量，但历史结果头没有保存精确请求向量、tokenizer revision 或 benchmark commit，因此每条均标记为 `mismatch_unresolved_historical_equivalence` 并附原因；不能把“版本漂移”当作已证明事实。
- 这 39 条中有 28 条原本满足 full-workload、23 条原本满足 physics 条件；为避免把未证实等价的历史输入当成可靠 X，它们分别从这两个 cohort 排除。结果侧 token 总量只用于核验，没有用于反向“校正”事前 X。
- Phase 1.6 已成功的 271 行保持原值与缓存向量一致。

因此，`x_workload_benchmark_*` 继续表示 benchmark `SampleRequest.prompt_len` 语义，而不是服务端实际上下文，也不是结果文件中的事后总量。

## Effective workload 审计

effective workload 可解析 492/694：GPQA 177、LMArena 256、Sourcegraph FIM 59；其中 28 条 GPQA 因上述历史等价性未证实，不进入可靠 full-workload cohort。

- 对 Chat 任务，使用固定 benchmark 消息构造、vLLM 0.11.1 归一化证据、固定 tokenizer/chat template，并使用 `add_generation_prompt=True`。
- 对 Sourcegraph FIM，使用官方固定 renderer；59 条 resolved run 的 effective 与 benchmark 长度一致。
- 132 条 Llama 因静态文件访问受限而 missing。
- 70 条 gpt-oss GPQA 因 vLLM 0.11.1 走 Harmony 路径、但历史容器解析到的 `openai-harmony` 版本未记录而 unresolved；未猜测实现。

在 492 条 resolved run 上，effective mean input length 相对 benchmark mean 的增量为：GPQA 中位数 10 token/request，LMArena 中位数 35.76，Sourcegraph FIM 为 0。该差值来自事前模板/消息格式，不使用结果侧输入 token 反推。

## Canonical 模型静态特征

27 个模型均保留一行：20 个具有已固定 config，7 个 Llama 具有公开 benchmark 元数据但 config 状态为 restricted。每个 canonical 字段均有长表记录 `source_field → canonical_field`、原值、变换、状态和原因。

已覆盖并人工抽查的结构包括 Qwen dense/GQA、Qwen MoE、DeepSeek MLA、Nemotron Mamba-Transformer hybrid、Gemma、gpt-oss 和 Llama 受限行。benchmark 实际 weight precision 优先于仓库运行 dtype；运行 dtype 只用于 KV byte proxy。结构上不适用的字段保留 null，不填零。

## Physics-derived 特征

physics 表保留全部 694 行，其中 406 行公式结果 available；进一步结合输入 reconciliation 后，383 行进入可靠 physics cohort。公式缺失原因分布为：

- 132：Llama static/tokenizer 访问受限；
- 70：gpt-oss Harmony effective workload unresolved；
- 63：Nemotron hybrid attention/SSM 结构不适用纯 Transformer 公式；
- 23：DeepSeek MLA 不适用当前标准 GQA KV/attention 公式。

可用行提供透明的 prefill projection、causal attention、MLP、decode output-cap upper bound、KV cache、weight bytes 及理想 per-GPU proxy。所有值均经过 finite/nonnegative 检查。它们是理论 proxy/上界，不是实测 FLOPs、显存峰值或实际生成工作量。

## 数据质量与泄漏检查

自动校验覆盖：

- 694/565 总量、唯一键和一对一/多对一连接基数；
- 五个 GPU Y 字段非空、有限、非负；
- eligibility 按逐行必需字段、有限/非负范围、reconciliation 状态独立重算，并与持久化 coverage 全分组对账；
- 27 模型 canonical 状态和受限原因；
- role manifest 与 master 全列一一对应；
- 精确 X allowlist + 运行后 denylist：实际输入/输出结果、throughput、ITL/latency、实际 batch、duration、能耗/功率、稳定性结果均不得进入 X，未知 `x_*` 字段默认拒绝；
- Git-aware 扫描覆盖全部已跟踪及未忽略文件（包括 `tests` 与被跟踪的 `data` 文件）；未发现持久化凭证、模型权重、raw timeline、Prometheus 数据或正式模型训练调用；
- Notebook 从头到尾执行，HTML 预览生成但按设计不进入 Git。

## RAPL zone 1 限定结论

限定检索检查了 694 份既有轻量结果头、固定 benchmark/Zeus 0.13.1 源码以及 Phase 1.6 审计材料，没有找到把 zone index 1 直接映射到 RAPL `name` 的字符串证据。因此 zone 1 仍只能称为 **platform-like diagnostic**。约 5.93 kW 的中位功率是异常边界线索，不能据此命名为 `psys`、CPU package 或整机功率；该结果不阻塞 GPU 主表。

## 建模可行性判断

下一阶段可以开始严格受控的 GPU 侧正式建模，但必须区分三个 cohort，不能用单一分数掩盖样本变化：

1. **主 baseline：** 565 条稳定 run，使用 base ex-ante X；
2. **workload sensitivity：** 381 条稳定、full-workload eligible run；
3. **physics sensitivity：** 305 条稳定、physics eligible run。

当前数据足以比较“任务事前特征 → GPU steady-state 能耗/平均功率”的基线与增量特征价值，但不足以证明跨模型家族、跨 GPU 或新架构的强泛化。配置网格存在重复请求表和同模型近邻点，下一阶段必须采用 group-aware split，并单独报告按 task、model family、GPU 的外推表现；不能随机拆分后宣称泛化。

## 下一阶段建议

1. 先固定本阶段 role manifest 和三个 eligibility cohort，不再从结果字段补 X。
2. 分别建立 energy 与 average power 的简单 baseline；保持 GPU 侧口径，duration 只作 Y/诊断关系，不进入 X。
3. 使用 model-family/config-group/GPU-aware 的分组切分，并做 leave-family/leave-GPU 压力测试。
4. 依次比较 base、full workload、physics 三组特征；报告 cohort 变化和置信区间，不只报告单一误差。
5. 对 output cap 只作上界情景；若未来需要真实生成工作量，必须在任务发起前另行提供计划输出长度，不得使用实际输出 token。
6. 若要补齐 Llama，等待同一官方仓库的合法授权；若要补齐 gpt-oss，必须拿到历史容器中 Harmony 版本/模板证据。
7. 若要继续 CPU/DRAM，先补采 RAPL `name`/path、socket/NUMA 拓扑和采集边界，另立课题，不混入 GPU Phase 2。

**停止点：** Phase 1.7 到此结束。未训练 Ridge、树模型、神经网络或任何其他正式预测模型。
