# 扩展到 565 条稳定样本后，GPU 功率仍可预测，但跨谱系与高功率外推仍未解决

## 执行摘要

- **推荐候选**：`RandomForest + P2-HardwareStatic + L1`。在 449 条开发样本、21 个模型的 model-group OOF 上，MAE 为 **272.4 W**、RMSE 为 **474.3 W**、R² 为 **0.941**；候选锁定后，在完全隔离的 116 条、6 个模型 Confirmatory holdout 上，MAE 为 **295.3 W**、RMSE 为 **424.6 W**、R² 为 **0.921**。
- **565 条全 stable cohort 应成为后续 GPU 功率主数据集**。P0/P1/P2 均完整覆盖 565 条记录、27 个 model IDs，117 条 Llama 全部恢复进入主模型；但该转向扩大的是覆盖范围，不等于已经解决未知谱系外推。
- **hardware scale 有用，但不是一个稳定的单变量结论**。aggregate rated power 对 CatBoost、Ridge 和 448 条 static 同支持集有改善；最终 RandomForest 的主要增益出现在 P1 → P2。P2-NoLabel 相对 P2 在 RandomForest 上恶化 11.5 W（95% model-cluster bootstrap：1.6 至 26.8 W），说明连续硬件量尚不能替代 GPU identity。
- **高功率系统性低估仍存在**。完整 565 条诊断 OOF 中，8-GPU MAE 为 720.0 W、signed bias 为 −223.9 W；top 10% MAE 为 1108.6 W、signed bias 为 −667.3 W。L2、pseudo-Huber 和温和加权均未同时改善总体误差与高功率误差。
- **最主要剩余问题是谱系外推，不是算法数量不足**。Leave-one-lineage-out 中 DeepSeek、Llama、Qwen 的 MAE 分别为 1461.3、695.9、362.0 W；在相同 task 内，DeepSeek、Llama、Qwen 仍明显困难。因此下一阶段应优先补齐困难谱系和高功率组合，而不应把本模型直接解释为跨 GPU 世代或节点功率模型。

## 1. 研究对象、口径与边界

本阶段只研究 ML.ENERGY Benchmark V3 的 **GPU 侧**、文本 LLM inference、稳定窗口数据，目标仅为 `y_gpu_avg_power_watts`：steady-state window 内全部 GPU 的聚合平均功率（W）。工况是 **high-load / near-saturation steady-state** serving，不代表任意请求到达率或低负载动态工况。

正式输入全部是执行前可知信息。`diag_model_id`、parent lineage、subfamily、task outcome、实际 token、吞吐、延迟、时长、利用率、温度以及任何目标派生量均未进入 X。模型谱系字段仅用于划分、覆盖统计和误差诊断；`diag_architecture_class` 表示核心网络结构，可以作为事前结构特征。产品/技术谱系与网络 architecture class 不是同一概念。

本阶段没有研究 energy/token、CPU/DRAM、node residual、Training、Diffusion、瞬态曲线或 arrival-rate 模型，也没有下载全量 raw timeline。

## 2. 数据覆盖：565 条全部进入 P0/P1/P2

主数据表包含 **565 条唯一 stable runs、27 个 model IDs**，P0-Core、P1-AggregatePower、P2-HardwareStatic 均为 565/565 完整。Llama 的 117 条记录全部进入主模型；未伪造 detailed static config、tokenizer 或 effective token。

| 诊断层 | 分组 | runs | models |
|---|---:|---:|---:|
| parent lineage | Qwen | 276 | 12 |
| parent lineage | Llama | 117 | 7 |
| parent lineage | GPT-OSS | 65 | 2 |
| parent lineage | Nemotron | 56 | 2 |
| parent lineage | Gemma | 29 | 2 |
| parent lineage | DeepSeek | 22 | 2 |
| architecture | dense_transformer | 203 | 10 |
| architecture | mixture_of_experts | 306 | 15 |
| architecture | hybrid_mamba_transformer | 56 | 2 |

Subfamily 覆盖为：Qwen3 226/9、Qwen3-Coder 50/3、Llama-3.1 69/4、Llama-3.3 19/1、Llama-4 29/2、GPT-OSS 65/2、Nemotron-Nano-v2 56/2、Gemma-3 29/2、DeepSeek 22/2（runs/models）。GPU 分布为 B200 291、H100 274；GPU count 为 1/2/4/8 卡分别 250/104/83/128 条；任务为 gpqa 187、lm-arena-chat 328、sourcegraph-fim 50 条。

硬件连续量沿用 Phase 3 冻结 reference proxy。必须特别说明：GPU SKU 映射仍为 ambiguous reference proxy；33 条 H100/MXFP4 记录没有冻结的原生 FP4 峰值，`precision_matched_peak_compute` 使用 H100 BF16 dense peak 作为明确标记的 fallback。因此不能把该字段解释为经过验证的 H100 FP4 物理峰值。

## 3. 隔离策略与评价协议

在任何 Phase 4 训练之前，按 parent lineage 分层并用稳定 hash（不读取 y）锁定 6 个模型作为 Confirmatory holdout，其余 21 个模型、449 条记录构成 development set。所有 feature bundle、算法、超参数、loss 和 weighting 决策只使用 development model-group OOF；候选锁定后才读取 Confirmatory 误差，且未据此调参。

开发 OOF、一次性 Confirmatory、完整 565 条诊断 OOF 是三种不同证据：前者用于选择，Confirmatory 用于独立评价，完整 OOF 只用于全样本切片和描述，不能重新参与选择。Parent-lineage、within-task lineage 与 architecture holdout 也在候选锁定和 Confirmatory 冻结之后，使用完整 565 条 cohort 作为 **post-lock diagnostic stress tests**；它们包含原 Confirmatory 模型，但不回流到候选、参数或 Confirmatory 指标。实际执行使用单独的 `full_cohort_*_folds.csv` manifests，不与 449 条 development manifests 混用。

## 4. 主模型比较：XGBoost 不再是冠军

下表为 449 条 development、21 个模型、完全相同 model-group folds 上的正式 P0/P1/P2 比较。P2-NoLabel 是单独 sensitivity，不用于主排名。

| Algorithm | Bundle | MAE W | RMSE W | R² | MdAPE % | sMAPE % |
|---|---|---:|---:|---:|---:|---:|
| DummyMean | P0 | 1625.9 | 2013.5 | -0.065 | 103.46 | 80.69 |
| DummyMean | P1 | 1625.9 | 2013.5 | -0.065 | 103.46 | 80.69 |
| DummyMean | P2 | 1625.9 | 2013.5 | -0.065 | 103.46 | 80.69 |
| Ridge | P0 | 454.8 | 738.3 | 0.857 | 17.54 | 22.58 |
| Ridge | P1 | 410.7 | 642.8 | 0.891 | 16.42 | 21.91 |
| Ridge | P2 | 396.8 | 629.0 | 0.896 | 15.70 | 21.22 |
| HistGradientBoosting | P0 | 288.4 | 557.1 | 0.918 | 10.03 | 14.68 |
| HistGradientBoosting | P1 | 284.1 | 513.0 | 0.931 | 10.16 | 13.86 |
| HistGradientBoosting | P2 | 294.9 | 540.1 | 0.923 | 10.16 | 13.94 |
| RandomForest | P0 | 298.0 | 628.2 | 0.896 | 9.18 | 14.70 |
| RandomForest | P1 | 309.6 | 571.1 | 0.914 | 11.12 | 14.44 |
| **RandomForest** | **P2** | **272.4** | **474.3** | **0.941** | **10.05** | **13.61** |
| CatBoost | P0 | 322.3 | 617.3 | 0.900 | 12.43 | 16.60 |
| CatBoost | P1 | 296.2 | 588.2 | 0.909 | 10.04 | 14.71 |
| CatBoost | P2 | 281.0 | 572.6 | 0.914 | 9.13 | 14.28 |
| XGBoost | P0 | 277.3 | 508.1 | 0.932 | 11.25 | 15.12 |
| XGBoost | P1 | 311.6 | 553.2 | 0.920 | 11.97 | 15.25 |
| XGBoost | P2 | 311.8 | 553.3 | 0.920 | 12.40 | 15.56 |

因此，565 条覆盖问题对应的开发集上，主候选是 RandomForest + P2，而不是 Phase 3 的 XGBoost。Phase 3 的 204.4 W 与这里的 272.4 W **不可直接同比**：Phase 3 只含 305 条 physics-complete 记录、模型组成与 folds 不同，且 204.4 W 是查看多个 outer-test 结果后的最低观察值，存在 winner selection bias。Phase 4 用更广谱系和独立 Confirmatory 评价换取了更可信的覆盖结论。

## 5. Hardware scale：有效，但主要来源依学习器而变

配对消融均使用相同的 449 行、21 个模型和相同 folds；区间为 1000 次 model-cluster bootstrap。

- **P0 → P1**：CatBoost ΔMAE = −26.0 W（95% CI −51.5 至 −4.8），Ridge = −44.1 W（−89.7 至 −7.6），说明 aggregate rated power 可提供可重复信号；但 RandomForest 为 +11.7 W、XGBoost 为 +34.3 W，区间均跨零，因此不是所有学习器都会受益。
- **P1 → P2**：最终 RandomForest ΔMAE = −37.2 W（−94.8 至 −1.6），完整静态硬件量对其有增益；CatBoost、Ridge 的点估计改善但区间跨零，XGBoost 基本不变。
- **P2 → P2-NoLabel**：RandomForest 去掉 GPU label 后 ΔMAE = +11.5 W（1.6 至 26.8），出现小但可辨识的恶化。只有 H100/B200 两个硬件支持点，不能据此声称连续硬件字段已经学会跨 GPU 世代外推。
- **Static subset**：在相同 448 行上，B1-Static、+aggregate rated power、+full HardwareStatic 的 MAE 分别为 189.7、176.9、183.0 W。这里 aggregate rated power 解释了大部分可见硬件增益，完整字段没有继续改善。

结论是：Phase 3 的 B4 改善相当一部分可解释为尺度工程，但不能归结为一个普适的 aggregate-rated-power 单变量效应；最终主候选还从其他 P2 字段或其与树模型的交互中获益。

## 6. 高功率偏差与 loss 实验

开发集上，RandomForest + P2 + L1 的总体 MAE 最低，为 272.4 W；top 25% MAE 为 682.4 W，top 10% MAE 为 1009.4 W，top 10% signed bias 为 −392.5 W。对同一候选，L2 将总体 MAE 提高到 285.2 W，top 10% MAE 提高到 1058.4 W；温和 high-power weighting 将总体 MAE 提高到 313.5 W，top 10% MAE 提高到 1091.4 W。XGBoost 的 L2、正确尺度化的 pseudo-Huber 和 weighting 也均不优于 L1。

完整 565 条诊断 OOF 进一步显示：B200 MAE 381.1 W、signed bias −60.9 W；H100 MAE 183.9 W、bias +65.7 W。8-GPU MAE 720.0 W、bias −223.9 W；top 25% MAE 643.8 W、bias −258.0 W；top 10% MAE 1108.6 W、bias −667.3 W。校准斜率整体约 0.885，top 10% 约 0.256，说明极高功率预测明显压缩。

因此，B200、8-GPU 和 top-power 样本仍有系统性低估；本阶段的损失函数调整没有把该问题缓解到可接受水平，也不存在“用普通功率区显著退化换来了高功率改善”的成功方案。

实现备注：XGBoost pseudo-Huber 的默认 1 W `huber_slope` 与数百至数千瓦标签尺度不匹配，会产生异常预测。本阶段把 delta 仅由每个训练 partition 的 y 中位数确定，并把该规则纳入 checkpoint 身份；修复后其 MAE 仍为 343.9 W，不改变候选选择。

## 7. 四层泛化必须分开解释

### 7.1 unseen model

锁定候选在 development model-group OOF 上 MAE 272.4 W；一次性 Confirmatory unseen-model MAE 295.3 W。两者相近，支持“对当前 ML.ENERGY 域中未见具体模型有可用信号”，但分谱系差异很大。

### 7.2 unseen parent lineage

在完整 565 条 cohort 上进行的 post-lock Leave-one-lineage-out，MAE（W）为：DeepSeek 1461.3、Llama 695.9、Qwen 362.0、GPT-OSS 268.4、Gemma 92.4、Nemotron 81.2。DeepSeek bias +1459.2 W，Llama bias −334.0 W，Qwen bias −259.8 W。跨谱系外推远弱于 model holdout，不能用一个 pooled “family MAE”掩盖这一差异，也不能把该锁定后诊断当成第二轮候选选择。

### 7.3 within-task unseen lineage

在至少覆盖 3 个 parent lineages 的 task 内，困难仍存在：gpqa 内 DeepSeek/Qwen MAE 为 1052.6/529.1 W；lm-arena-chat 内 Llama/DeepSeek/Qwen MAE 为 832.5/397.2/285.7 W。GPT-OSS 只出现在 gpqa，Gemma 和 Llama 只出现在 lm-arena-chat，相关结果已标为 task-lineage confounded。由于在相同 task 内仍出现明显崩溃，失败不能只归因于 task/domain 混杂；但混杂确实限制了对 GPT-OSS、Gemma、Llama 的因果式谱系解释。

### 7.4 unseen architecture class

探索性 holdout MAE 为 dense 313.8 W、MoE 558.4 W、hybrid 81.2 W。hybrid 仅由 Nemotron 支撑，MoE 也与若干谱系高度相关，所以这是 architecture stress test，不是稳定的跨架构平均性能。它说明 MoE 外推最困难，但不能把差异完全归因于结构本身。

重新纳入 Llama、GPT-OSS、DeepSeek、Nemotron 明显改善了覆盖和问题识别，却没有证明跨 parent lineage 泛化已实质解决。

## 8. Confirmatory 独立评价

候选锁定后仅评价一次的 held-out model IDs 为：

1. `Qwen/Qwen3-235B-A22B-Thinking-2507`
2. `deepseek-ai/DeepSeek-R1-0528`
3. `google/gemma-3-27b-it`
4. `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`
5. `nvidia/NVIDIA-Nemotron-Nano-9B-v2`
6. `openai/gpt-oss-120b`

总体 116 行：**MAE 295.3 W、RMSE 424.6 W、R² 0.921、MdAPE 16.46%、sMAPE 19.58%、signed bias +187.7 W**。分 lineage MAE/bias（W）为：Qwen 236.4/+57.9、DeepSeek 534.7/−478.5、Gemma 81.3/−61.5、Llama 973.3/+973.3、Nemotron 72.2/+6.3、GPT-OSS 388.9/+388.9。

GPU slice：B200 60 行，MAE 293.4 W、bias +100.9 W；H100 56 行，MAE 297.3 W、bias +280.8 W。GPU-count slice 的 MAE/bias（W）为：1 卡 174.9/+105.5，2 卡 295.8/+277.5，4 卡 133.5/+107.2，8 卡 572.6/+260.2。

Confirmatory 总体结果支持候选在当前域中的独立可复现性能，但单模型 lineage slice 不能估计 lineage 内部方差。尤其 Llama held-out 模型出现严重过预测，说明“平均 295.3 W”不能替代谱系边界说明。

## 9. 八个核心问题的直接回答

1. **565 条后的 model-holdout 性能如何？** development OOF MAE 272.4 W；独立 Confirmatory MAE 295.3 W；完整 565 条诊断 OOF MAE 285.4 W。
2. **Hardware scale 在全样本上仍有效吗？** 有效但依学习器而异；CatBoost/Ridge 的 P0 → P1 明确改善，最终 RandomForest 的 P1 → P2 明确改善。
3. **B4 增益来自哪里？** aggregate rated power 解释了 static subset 和部分学习器的大部分增益；最终 RandomForest 还需要完整 P2。不能把 B4 全部归因于单一尺度变量。
4. **重新纳入谱系后泛化改善吗？** unseen-model 指标可用且覆盖更完整，但 unseen parent lineage 和 within-task unseen lineage 仍对 DeepSeek/Llama/Qwen 明显失败。
5. **family 失败是否只是 task 混杂？** 不是。混杂存在并已标注，但相同 task 内仍有较大谱系误差。
6. **XGBoost 仍是主模型吗？** 不是；当前锁定的是 RandomForest + P2 + L1。
7. **高功率低估被 loss 缓解了吗？** 没有；L2、pseudo-Huber 和 weighting 均未同时改善总体与高功率区。
8. **能否锁定候选并独立评价？** 可以；候选已在读取 Confirmatory 误差前锁定，并获得一次性 116 行评价，但只能在下述适用边界内使用。

## 10. 推荐候选、适用边界与部署解释

推荐候选为 `RandomForest + P2-HardwareStatic + L1`，最终超参数为 400 trees、`max_depth=12`、`min_samples_leaf=2`、`max_features=0.7`。它适用于 ML.ENERGY 域内 H100/B200、稳定、近饱和、高负载的 GPU 侧文本 LLM inference，预测的是聚合 GPU 平均功率而非节点功率。

其适用边界如下：

- 只在 H100/B200 两个标签和冻结 reference proxy 上验证，不支持跨 GPU 世代声明；
- 不适用于低负载、动态 arrival rate、瞬态、训练或扩散任务；
- 不预测 CPU/DRAM/node power，也不应把 GPU 功率直接当作节点功率；
- 对 DeepSeek/Llama/Qwen 的未见谱系和 8-GPU/top-power 组合存在明显风险；
- P2 含 categorical GPU label，连续硬件表示尚未替代 identity；
- Confirmatory 每个 lineage 只有一个 model，不能据此估计谱系内不确定性。

## 11. 决策与下一阶段建议

1. **数据主线**：正式从 305 条 Physics cohort 转向 565 条 stable cohort 做 GPU 功率主模型；305/448 条子集保留为特征机制诊断，不再限制主覆盖。
2. **候选锁定**：保留 RandomForest + P2 + L1 作为 Phase 4 候选基线；不要因 Confirmatory slice 再回调参数。
3. **下一步优先级**：继续优化功率模型所依赖的数据覆盖，而不是继续增加通用算法。优先采集或补强 DeepSeek、Llama、Qwen 的多 task、多 GPU-count，尤其 B200/8-GPU/top-power 组合，并设计新的、仍然纯事前的容量/并行机制特征。
4. **节点框架**：现阶段可并行设计接口和误差传播方案，但不建议把该候选直接接成正式节点功率模型。应先设立谱系与高功率外推验收线，再进入 CPU/DRAM/node residual 的独立阶段。

## 12. 可复现性与审计证据

表、folds、逐行 OOF/Confirmatory 预测、候选模型、哈希回执、12 张图和执行后 notebook 均保存在 `step6/`。独立验证器不导入建模 helper，重新核对冻结输入、数据边界、split 隔离、逐行指标、配对 bootstrap、一次性 Confirmatory 哈希、图表与密钥模式扫描。环境和训练 provenance 见 `analysis/training_provenance.json`；完整验证结果见 `reports/verification_receipt.json`。

## 13. 关键限制

- 这是单一 benchmark 的观察性数据；性能不能自动外推到其他推理引擎、时钟/功率限制、批处理策略或数据中心。
- parent lineage、task、architecture 和 GPU composition 仍有相关性；stress tests 不能当作单因素因果估计。
- bootstrap 量化的是当前模型簇重采样不确定性，不包含数据集迁移、硬件 reference proxy 错配或候选选择的全部不确定性。
- full-cohort OOF 是锁定后诊断，不是第二次无偏模型选择结果。
- 报告中的负 signed bias 表示预测偏低，正值表示预测偏高。

## 14. 最终结论

Phase 4 已完成“覆盖面定型”而未完成“跨域外推定型”：565 条 stable runs 能支持一个明确、纯事前、GPU 侧高负载稳态功率候选，并用一次性 Confirmatory 给出约 **295 W MAE** 的独立评价；然而谱系留出和极高功率样本仍暴露出数百至上千瓦级误差。下一阶段最有价值的工作是针对困难谱系与高功率组合补数据、补机制特征并设立边界验收，而不是直接宣称通用 GPU 或节点功率预测已经完成。
