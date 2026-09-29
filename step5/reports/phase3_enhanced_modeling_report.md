# ML.ENERGY Phase 3.0：高负载 GPU 功率/能耗增强预测与硬件物理建模

## 摘要结论

本阶段已在冻结的 ML.ENERGY Benchmark V3 数据上完成预定实验，并在此停止。研究对象严格限定为 **GPU 侧、文本 LLM inference、high-load steady-state / near-saturation** 工况；所有正式输入均为执行前可知变量。

120 个配对实验中，最低的已观测功率误差来自 **B4-Hardware + XGBoost**：在 305 条 Physics cohort、冻结 model-holdout 五折上，MAE 为 **204.4 W**，RMSE 为 **370.6 W**，R² 为 **0.960**，MdAPE 为 **8.6%**。但该组合是在查看多组 outer-test 结果后继续深入分析的，属于**探索性候选**，并非预注册冠军；204.4 W 及其普通 bootstrap 区间没有校正**获胜者选择偏差**，不得解释为无偏的最终模型性能。

相对历史控制 B0-Core + HGB（253.3 W），该探索性候选的行加权 MAE 观察差为 −48.9 W。逐模型等权的配对差及 bootstrap draw 已写入机器可读表，但它们同样不消除结果知情的候选选择，因此只用于描述，不作确认性显著性声明。

更完整的跨学习器结果显示：B3→B4 的行加权 MAE 在 10 个学习器中 8 个改善，其中 Ridge、ElasticNet、SplineGAM、ExtraTrees 的等模型区间低于 0；XGBoost 改善 33.5 W，但区间跨 0。B4→B5 则在 10 个学习器中 9 个恶化。因此连续硬件尺度呈现**有希望但学习器相关的观察增益**，不是稳定的独立增量证据；当前 workload/hardware 比率总体无益。硬件仅有 H100/B200 两个标签，不能据此宣称已建立跨 GPU 世代外推的物理规律。

泛化难度决定了实际误差：探索性 B4+XGBoost 在 10 次 Random 80/20 上平均 MAE **97.2±23.6 W**，Config holdout 为 **94.7 W**，Model holdout 为 **204.4 W**，Family holdout 升至 **736.7 W**。这些观察提示同分布与未知模型预测具有可行性，但不能作为可靠未知家族外推或生产部署的确认性证据。

Energy/token 的最低观测主矩阵误差为 B1-Static + XGBoost：MAE **0.620 J/token**、R² **0.677**。raw 方案出现 1 条负预测（最小 −0.0207 J/token）；log1p 方案在本次运行中未观察到负预测，但 MAE 略差至 **0.650 J/token**。`expm1` 并不从数学上保证预测非负，所以本报告不再称其为“正值约束”。

## 1. 研究目标与工况定位

本阶段回答五个问题：算法上限、B3 物理特征为何未稳定提升、硬件物理表示是否有价值、误差如何随泛化难度退化，以及模型适用边界。

565 条稳定样本的运行后诊断显示，平均 batch / `max_num_seqs` 中位数为 **0.9983**，p5–p95 为 **0.9422–0.9999**。因此当前数据应表述为高负载、近饱和的稳态服务工况。

这些 batch 与吞吐信息只用于工况刻画和误差切片，未进入任何 B0–B5 输入。当前研究不是任意请求到达率下的动态 `P(λ)` 建模，也不是瞬态功率建模。

## 2. 数据、cohort 与输入边界

冻结 cohort 如下：

| Cohort | 样本数 | 用途 |
|---|---:|---|
| Base | 565 | 基础事前特征与历史控制 |
| Static | 448 | 静态模型结构完整 |
| Full | 381 | 有效工作量完整 |
| Physics | 305 | B3–B5 与本阶段严格配对实验 |

核心矩阵中的所有 B0/B1/B2b/B3/B4/B5 比较均使用相同 305 条 run、相同 run_key、相同 model folds。矩阵覆盖 2 个目标 × 6 个 bundle × 10 类算法，共 120 个实验、36,600 条 OOF 预测。超参数选择只发生在 outer-train 内的 group-aware inner CV；outer-test 结果不参与折内超参数选择，但跨 120 个结果观察最低值仍产生候选选择偏差。

禁止进入 X 的字段包括 actual output tokens、实际 batch、throughput、latency、duration、GPU utilization、温度、时钟、实测带宽及目标本身。模型 ID、质量标记和重复编号也不作为输入。

## 3. B0–B5 与 GPU hardware profile

- B0-Core：GPU 型号/数量、并行配置、max_num_seqs、参数量和精度。
- B1-Static：层数、宽度、attention/KV heads、上下文和结构类型。
- B2b-Effective：effective input-length 分布。
- B3-Physics：理论 FLOPs、KV bytes、理想权重 bytes 等 workload/model physics。
- B4-Hardware：在 B3 上加入峰值计算、HBM 带宽/容量、额定功率、互联带宽及 aggregate rated power。
- B5-HardwarePhysics：再加入 compute-time、memory-time、容量占比及 compute/memory pressure 等理论代理。

公开 ML.ENERGY 标签仅给出 H100/B200，不能唯一确定具体 SKU。本阶段采用官方资料支持的 H100 SXM 80GB 与 B200 SXM 180GB 参考代理，并将映射状态均标为 `ambiguous`。峰值计算使用 dense 口径；所有时间量均称为 theoretical lower-bound proxy，不称为实测执行时间。来源与解释逐字段保存在 `gpu_hardware_specs.json` 和 `hardware_source_manifest.json`，主要依据 [ML.ENERGY 数据卡](https://huggingface.co/datasets/ml-energy/benchmark-v3)、[NVIDIA H100 规格](https://www.nvidia.com/en-gb/data-center/h100/)、[NVIDIA HGX 参考架构](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html) 与 [HGX B200 PCF summary](https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf)。

## 4. Physics coverage / identifiability

### 观测

- B3/B5 连续变量并非静态，多数具有约 1–2 个 log10 数量级的范围。
- B5 中 KV capacity ratio 的动态范围最大，约 2.1–2.2 log10；B3 prefill attention FLOPs 约 2.10 log10。
- 纯硬件静态字段只有两个取值，且相互完全秩相关，因为仅有 H100/B200 两个硬件点。
- 26 个物理数值特征在 log1p+标准化 PCA 中，PC1 解释 41.9%，前 3 个解释 82.1%，前 5 个解释 95.8%。
- 冻结 Phase 2 B2b-HGB 残差与部分 B3 特征存在中等相关，例如 decode 侧 p90 指标 Spearman 约 0.41；这提示残差中仍有可用结构，但不构成因果证据。

### 解释

Physics 失败不能归结为“所有物理量都没有变化”。更合理的解释是：多个理论代理高度相关、有效维度较低；硬件静态维度又只有两个支持点。学习器可能利用其中一部分，但难以识别稳定且可外推的独立效应。

### 限制

PCA 描述的是 log1p+标准化后的几何结构，不代表原始物理量的线性主成分；残差相关也不证明特征具有因果增量。

## 5. RQ1：算法与预测精度上限

核心矩阵的前列方案如下；B4 已对全部 10 个学习器和两个目标完整运行。

| 方案 | Power MAE (W) | RMSE (W) | R² | MdAPE |
|---|---:|---:|---:|---:|
| **B4-Hardware + XGBoost** | **204.4** | **370.6** | **0.960** | **8.6%** |
| B1-Static + XGBoost | 237.9 | 391.1 | 0.956 | 11.3% |
| B3-Physics + XGBoost | 237.9 | 379.6 | 0.958 | 10.6% |
| B1-Static + CatBoost | 238.0 | 358.1 | 0.963 | 11.0% |
| B2b-Effective + CatBoost | 243.8 | 372.6 | 0.960 | 12.7% |
| B0-Core + HGB（历史控制） | 253.3 | 423.1 | 0.948 | 11.9% |

### 结论

HGB 并非现有数据与特征下的唯一合理算法；XGBoost 在 B4 上给出最低观察误差。不过，本研究没有额外的最外层选择—评估层，不能用同一批 outer-test 排名后再把最低值当作无偏“算法上限”。因此“204.4 W”只回答探索性 RQ1：它是本矩阵的最低观察值，不是确认性冠军性能、不可突破的理论上限，也不能外推到新硬件或生产流量。

## 6. RQ2：B3 Physics 的跨学习器增量

在完全相同 305 条样本和 folds 上，B2b→B3 的 Power MAE 变化为：

| 模型 | ΔMAE，B3−B2b (W) | 方向 |
|---|---:|---|
| XGBoost | −16.8 | 改善，但等模型 bootstrap 区间跨 0 |
| HGB | +14.5 | 变差 |
| ExtraTrees | +31.0 | 变差 |
| RandomForest | +37.8 | 变差 |
| CatBoost | +54.7 | 区间支持恶化 |
| ElasticNet | +85.6 | 变差 |
| Ridge | +122.5 | 变差 |
| SmallMLP | +786.8 | 严重不稳定 |

### 结论

B3 不是完全无信息：XGBoost 能提取约 16.8 W 的行加权增益。但增益没有跨模型簇形成稳定证据，而大多数学习器恶化。因此 Phase 2 的负结果既与学习器交互有关，也与特征冗余、低维结构和有限覆盖有关；不能简单解释为“换成更强模型就能解决”。

## 7. RQ3：硬件物理表示与目标策略

| XGBoost bundle / strategy | Power MAE (W) | R² | 解释 |
|---|---:|---:|---|
| B3-Physics | 237.9 | 0.958 | workload/model physics + GPU label |
| **B4-Hardware** | **204.4** | **0.960** | 加连续硬件静态量 |
| B5-HardwarePhysics | 241.7 | 0.943 | 再加归一化 pressure proxies |
| B5-NoLabel | 234.0 | 0.947 | 去掉类别 GPU label |
| B4 + power_fraction target | 239.8 | 0.950 | 以 aggregate rated power 归一化目标 |

### 配对不确定性（探索性）

- B3→B4：行加权改善 33.5 W；等模型改善 24.8 W，95% 区间 **−10.0 至 55.5 W 改善**，跨 0。
- B4→B5：行加权恶化 37.4 W；等模型恶化 49.9 W，95% 区间 **11.0–105.7 W**，支持 B5 当前构造对该探索性候选有害。
- B5 去除 GPU label：行加权改善 7.8 W；等模型差异区间跨 0，未证明连续物理量可稳定替代身份标签。

### 跨学习器结果、解释与限制

B3→B4 在 Ridge、ElasticNet、SplineGAM、ExtraTrees 上呈现区间不跨 0 的等模型改善，在 XGBoost、HGB、RF、MLP 上方向改善但区间跨 0，在 CatBoost 上恶化，Dummy 不变。由此更准确的表述是：连续硬件字段在若干学习器上显示增量信号，但效果并不普遍稳定。对结果知情的 B4+XGBoost 深入分析中，aggregate rated power、GPU count 和参数量在 fold-local permutation 中最重要；但这些变量与 GPU 标签高度相关，且只有两个 GPU 点。它更接近“更好的两类硬件表征”，不是已经学到可外推的新 GPU 物理规律。

B5 恶化说明理论 lower-bound 比率可能放大噪声、引入冗余或与树模型的有效划分不匹配。power_fraction 也没有优于直接预测，额定功率不应被当作自动有效的归一化尺度。

## 8. RQ4：Random → Config → Model → Family 泛化退化

| 方案 | Random MAE (W) | Config | Model | Family |
|---|---:|---:|---:|---:|
| B0-Core + HGB | 130.0±24.9 | 132.8 | 253.3 | 672.0 |
| **B4-Hardware + XGBoost** | **97.2±23.6** | **94.7** | **204.4** | **736.7** |

### 观测

B4+XGBoost 在 Random、Config 和 Model 三个层级均优于历史控制，但在 Family holdout 上反而更差 64.7 W。Family R² 仅 0.590，而 model-holdout R² 为 0.960。

### Config 层级的识别限制

Physics cohort 的 Config split 有 303 个 config groups / 305 条 run，其中 **301 个单例**，只有 4 条 run 属于重复配置组。因此 Config holdout 几乎等同 run-level 划分，和 Random 的接近不能当作强独立证据；真正有区分度的是 Model 与 Family 层级。

### 解释

增强模型提升了插值与未知模型预测，但没有提升跨家族能力。Family holdout 图显示明显的分段系统偏差：某些高功率家族被严重低估，另一些被高估。当前结构变量和两点硬件信息无法替代家族覆盖。

## 9. Measured vs Predicted、残差和误差切片

Model-holdout 下，高功率 B200 点仍存在系统性低估，残差绝对值随功率升高而扩张，提示异方差。Family holdout 的分段偏差更强，说明失败并非少数随机噪声。

按 post-run load ratio 切片时，近饱和主群（ratio≥0.99）占 264/305；它是样本主体。低于 0.95 的 23 条样本误差更高，但该字段是运行后诊断，不能用于事前路由或作为模型输入。

关键图：

- [Random train/test](figures/actual_vs_predicted_random_best.png)
- [Model holdout OOF](figures/actual_vs_predicted_model_holdout_best.png)
- [Family holdout OOF](figures/actual_vs_predicted_family_holdout_best.png)
- [Power residuals](figures/residuals_model_holdout_best.png)
- [Energy/token OOF](figures/actual_vs_predicted_energy_best.png)
- [Energy/token residuals](figures/residuals_energy_best.png)

## 10. Energy/token 长尾与输出值域

| 策略 | MAE (J/token) | RMSE | R² | 负预测数 |
|---|---:|---:|---:|---:|
| B1 + XGBoost raw | **0.620** | 2.114 | 0.677 | 1 |
| B1 + XGBoost log1p | 0.650 | 2.113 | 0.677 | 0 |

Energy/token 的误差受少数高值样本主导。残差图显示 10–34 J/token 区域普遍低估。本次 log1p 运行没有出现负预测，但这只是观察结果，并非由 `expm1` 保证的正值约束；同时它未改善 MAE。本报告没有通过事后截零来美化主结果。若进入部署阶段，应单独预注册真正的正链接/正分布模型和评价标准。

## 11. 解释性结果

对 B4+XGBoost 与 B3+XGBoost 完成了折内置换重要性；对探索性 B4+XGBoost 候选完成 5 个关键连续变量的 fold-local PDP。SHAP 在冻结环境不可用，未做 SHAP 专属归因。

B4 的平均重要性顺序以 aggregate rated power、GPU count、total params、max_num_seqs 为主。PDP 的五条外折曲线水平明显分离，说明全局趋势受到留出模型组成强烈影响。解释性结论应表述为“模型在本数据上的依赖结构”，不能表述为因果功率定律。

## 12. RQ1–RQ5 最终回答

- **RQ1：** 最低观察值为 B4-Hardware + XGBoost，model-holdout MAE 204.4 W；但多候选 outer-test 排名产生获胜者选择偏差，本阶段不能给出无偏的算法上限。
- **RQ2：** B3 对 XGBoost 有弱增益，对多数模型无效或有害；原因是学习器交互、共线/低维表示和有限覆盖共同作用，而非单一因素。
- **RQ3：** B4 在多数学习器上方向改善、在部分学习器上区间不跨 0，但 XGBoost 的配对区间跨 0且候选分析为结果知情；结论是“有希望的观察信号”，不是稳定增量证据。当前 B5 归一化代理总体无益，两硬件点不足以证明跨 GPU 通用性。
- **RQ4：** Random/近似 run-level 的 Config/Model 尚可，Family 明显失效；Config 的 303/305 组结构不足以构成强独立层级，增强模型也没有解决跨家族泛化。
- **RQ5：** 模型仅支持本 benchmark 的高负载稳态 GPU 侧文本推理。不能支持任意 λ、瞬态、未知 GPU 世代、生产流量或其它任务域。

## 13. 不确定性与主要限制

1. Physics cohort 只有 305 条、14 个 model IDs，cluster bootstrap 区间较宽。
2. 只有 H100/B200 两个硬件标签，且具体 SKU 未由公开数据唯一确定；硬件字段是最佳证据代理。
3. B4 的若干硬件量彼此高度相关，无法分离单个物理量的独立效应。
4. Family holdout 只有 3 个有效家族折，压力测试方差高但失败幅度很大。
5. 任务是高负载稳态聚合 GPU 功率，不覆盖请求到达过程、瞬态或单请求能量轨迹。
6. Energy/token 长尾和极端点仍未解决；正值域与误差最优之间存在权衡。
7. 第三方 SHAP 不可用，解释性采用 permutation/PDP；这不影响预测指标，但限制局部归因形式。
8. 120 个 outer-test 结果被共同查看，B4+XGBoost 后续分析是结果知情的；没有最外层选择—评估层，因此不报告经获胜者选择校正的冠军区间。

## 14. 下一阶段建议

下一阶段应先审查本报告，再决定是否立项。优先级建议：

1. 扩充 GPU 型号与明确 SKU，使硬件物理量至少有 4–6 个独立支持点；这是验证 B4 可迁移性的首要条件。
2. 扩充模型家族，尤其是当前 Family holdout 系统失败的高功率家族；预注册跨家族评价，不以 random split 替代。
3. 重构 B5：减少共线 proxy，区分 compute、HBM capacity、HBM bandwidth 与并行通信，先做符号/量纲审计再训练。
4. 为 energy/token 预注册正值输出模型和长尾损失，保留 raw 指标对照，不做未标注 clipping。
5. 若研究目标转向真实调度，另建包含请求到达率和瞬态 telemetry 的数据集，再研究动态 `P(λ)`；不得从本阶段样本反推。

不建议下一阶段仅继续堆叠更多算法或扩大无约束超参数搜索；当前主要瓶颈是家族与硬件覆盖，而不是缺少第 11 个回归器。

## 15. 停止条件与复现入口

Phase 3.0 到此停止，不扩展到 Training、Diffusion、CPU、DRAM、Node Residual、全量 raw timeline 或正式生产部署。

复现命令与文件说明见 [step5/README.md](../README.md)。可执行证据总览见 [phase3_enhanced_modeling.ipynb](../notebooks/phase3_enhanced_modeling.ipynb)，机器可读结果位于 `step5/analysis/` 与 `step5/reports/*.csv`。
