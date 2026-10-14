# Phase 5：事前信息完整度与总功率/平均单GPU目标的预测比较

## 执行摘要

所有主矩阵、相同支持集重跑和敏感性单元共224组，均采用5-fold model-group outer CV、3-fold model-group inner CV，算法独立调参。最低信息L0的最低观察OOF MAE为 **283.5 W**（CatBoost、T-total），per-GPU MAE为 **98.1 W/GPU**。这是多个候选中最低观察值，有winner-selection bias，不是新的独立最终性能。

原565条stable records、27 models完整保留；详细结构和workload支持集按真实字段重算，physics和effective的缺失边界保留。既有Phase4 Confirmatory已经被查看，本阶段所有结果均是nested OOF或明确标记的post-hoc诊断。正式对象仍为GPU侧、文本LLM inference、stable、high-load / near-saturation steady-state。

## RQ1：最低信息模型的能力

L0只要求GPU型号/卡数、总参数量、架构类别和benchmark precision；其硬件lookup及weight bytes自动派生不增加额外用户输入负担。L0/L0A/L1主支持集均565条。按相同L0支持集的算法独立nested OOF：最低MAE 283.5 W，RMSE 509.8 W，R² 0.926，MdAPE 12.54%，sMAPE 16.11%。各算法独立调参，inner CV不混合竞争；本阶段每个非基线算法统一8候选，较默认24候选缩减，搜索覆盖有限。

## RQ2：active params 与部署信息的独立价值

L0→L0A：在相同支持集（565行）上，16组 algorithm×target 比较的ΔMAE范围为 -25.4 至 11.7 W；2组95%模型簇区间全部低于0，0组全部高于0，其余跨0。完整明细包括row-weighted和equal-model口径，见 feature_increment_effects.csv。

L0A→L1：在相同支持集（565行）上，16组 algorithm×target 比较的ΔMAE范围为 -91.2 至 33.6 W；3组95%模型簇区间全部低于0，0组全部高于0，其余跨0。完整明细包括row-weighted和equal-model口径，见 feature_increment_effects.csv。

这些比较均565-row support、完全相同run keys/outer folds，不能把多个学习器结果平均成一个普适“物理增益”。model_parallel=num_gpus/DP先校验整数；weight residency是理想分配代理，不描述EP跨DP共享expert的实际显存分布。max_num_seqs_per_gpu仅为明确的尺度代理，未假设per-replica配置scope。

## RQ3：详细static结构的matched-support增益

在相同支持集（448行）上，16组 algorithm×target 比较的ΔMAE范围为 -77.6 至 11.2 W；0组95%模型簇区间全部低于0，0组全部高于0，其余跨0。完整明细包括row-weighted和equal-model口径，见 feature_increment_effects.csv。

L1@L2-support已重新训练；没有拿L1@565与L2较小cohort的绝对MAE相减。Hybrid的dense/MoE transformer layer composition为结构N/A并带applicability标记，特殊layer pattern仅进入L2X；unknown仍缺失并有indicator。Llama未伪造层数、KV heads或tokenizer。L2X扩展的增量同样在严格共同支持集判断。

## RQ4：workload独立增量与benchmark/effective

L2→L3：在相同支持集（448行）上，16组 algorithm×target 比较的ΔMAE范围为 -60.5 至 26.2 W；0组95%模型簇区间全部低于0，0组全部高于0，其余跨0。完整明细包括row-weighted和equal-model口径，见 feature_increment_effects.csv。

Benchmark→effective：在相同支持集（383行）上，16组 algorithm×target 比较的ΔMAE范围为 -55.9 至 21.0 W；0组95%模型簇区间全部低于0，1组全部高于0，其余跨0。完整明细包括row-weighted和equal-model口径，见 feature_increment_effects.csv。

后者在effective可靠可用的相同support/folds上比较；effective不自动被解释成更高阶。L3仅使用intensity统计与max_output_tokens，request repeats、总请求和total tokens未进入主层。Tokenizer-free characters/UTF-8 bytes通过全部三类冻结请求哈希验证，覆盖565，无completion或tokenizer；L1T可作为探索代理，但任务/domain模式和模板影响仍可能混杂。

## RQ5：compact physics是否增加独立信息

在相同支持集（305行）上，16组 algorithm×target 比较的ΔMAE范围为 -50.8 至 16.1 W；2组95%模型簇区间全部低于0，0组全部高于0，其余跨0。完整明细包括row-weighted和equal-model口径，见 feature_increment_effects.csv。

精简代理来自benchmark mean/second moment/p95/max与冻结静态结构；MLA/hybrid不套GQA公式，运行KV dtype未知不伪造。总prefill FLOPs是三块确定性加和，保留审计值但从正式X去掉；weight residency已经在L1，不重复添加。full-square attention约定继承旧研究，忽略sliding window和真实kernel细节。physics支持仅Qwen/Gemma，不能推断跨所有谱系稳定；旧B5比率仅sensitivity。公式与边界见physics_feature_formulas_phase5.md。

L4为L3已有事前信息的确定性变换，不新增独立观测信息；它检验的是表示方式对有限学习器的价值，并非新增物理测量或因果机制。当前多数算法×target区间跨0，少数得到支持，不能称为跨算法稳定的普适收益。

## RQ6：G-perGPU是否改善卡数尺度稳定性

同算法/level/cohort的paired comparison中，ΔMAE定义为G恢复total后减T-total；主层相关比较的点估计范围为 -1188.3 至 109.7 W，9组的模型簇95%CI全部低于0。需逐algorithm×level解释，不能从低per-GPU误差自动推断total更准。G的inner和final-group CV都以恢复后total MAE选参数；num_gpus继续在X中。

L1工程候选的卡数切片（XGBoost / G-perGPU）：

| slice_value | n_rows | mae | per_gpu_mae | signed_bias | per_gpu_signed_bias |
|---|---|---|---|---|---|
| 1 | 250 | 118.413 | 118.413 | 5.892 | 5.892 |
| 2 | 104 | 112.041 | 56.021 | -29.002 | -14.501 |
| 4 | 83 | 195.198 | 48.800 | 30.138 | 7.535 |
| 8 | 128 | 445.838 | 55.730 | 87.536 | 10.942 |

同一算法的T/G卡数切片见gpu_count_scaling_source.csv；8卡total误差需要结合W/GPU解释。平均单GPU口径指aggregate/卡数，不是逐卡独立测量。本矩阵负功率预测计数（跨实验重复support，非unique run数）为58，保持原值没有静默clip；逐算法和slice可查prediction原始输出。

修正后负值仅出现在L2X sensitivity的Ridge/ElasticNet T-total基线，各29个，集中于outer fold0的两个Gemma模型；主层及所有G-perGPU单元均无负值。这些线性基线没有非负约束，反映扩展静态组合的外推风险，不是实测负功率。没有据此改参数或删记录；范围、数值和保存OOF来源见`negative_prediction_audit.md`，它们未进入研发候选。

卡数切片显示的是误差幅度/偏差，不能单凭MAE曲线证明条件残差方差已稳定：GPU count同时与GPU型号、模型、并行配置混杂。平均单GPU误差缩小仍须结合恢复total后的paired区间与校准判断，不作为因果消除异方差的证明。

## RQ7：不同算法在哪些信息层适配

各层按自身support最低观察值：

| feature_level | target_strategy | algorithm | n_rows | n_models | mae | rmse | r2 | per_gpu_mae |
|---|---|---|---|---|---|---|---|---|
| L0-Basic | G-perGPU | HistGradientBoosting | 565 | 27 | 289.978 | 504.766 | 0.928 | 98.278 |
| L0-Basic | T-total | CatBoost | 565 | 27 | 283.519 | 509.823 | 0.926 | 98.112 |
| L0A-ActiveAware | G-perGPU | CatBoost | 565 | 27 | 269.846 | 454.445 | 0.941 | 98.344 |
| L0A-ActiveAware | T-total | RandomForest | 565 | 27 | 291.174 | 514.226 | 0.925 | 101.845 |
| L1-Deployment | G-perGPU | XGBoost | 565 | 27 | 202.698 | 322.671 | 0.970 | 82.501 |
| L1-Deployment | T-total | XGBoost | 565 | 27 | 281.165 | 541.243 | 0.917 | 112.634 |
| L2-StaticCore | G-perGPU | HistGradientBoosting | 448 | 20 | 162.991 | 241.744 | 0.983 | 82.193 |
| L2-StaticCore | T-total | HistGradientBoosting | 448 | 20 | 205.311 | 391.109 | 0.955 | 88.021 |
| L3-Workload | G-perGPU | XGBoost | 448 | 20 | 172.646 | 285.310 | 0.976 | 72.776 |
| L3-Workload | T-total | XGBoost | 448 | 20 | 177.683 | 319.855 | 0.970 | 68.175 |
| L4-CompactPhysics | G-perGPU | XGBoost | 305 | 14 | 183.111 | 293.201 | 0.975 | 67.307 |
| L4-CompactPhysics | T-total | XGBoost | 305 | 14 | 185.233 | 362.211 | 0.962 | 62.478 |

Information-Accuracy Curve的高层support不同，只描绘各信息要求下工程可达性能；特征增益以matched-support effects为准。完整8算法×6主层×2目标明细：

| feature_level | algorithm | target_strategy | n_rows | mae | rmse | r2 | mdape | smape | signed_bias | per_gpu_mae |
|---|---|---|---|---|---|---|---|---|---|---|
| L0-Basic | CatBoost | G-perGPU | 565 | 294.068 | 500.026 | 0.929 | 12.626 | 16.358 | -40.977 | 99.528 |
| L0-Basic | CatBoost | T-total | 565 | 283.519 | 509.823 | 0.926 | 12.544 | 16.110 | -9.792 | 98.112 |
| L0-Basic | DummyMean | G-perGPU | 565 | 510.852 | 822.627 | 0.808 | 22.463 | 23.702 | 9.240 | 147.822 |
| L0-Basic | DummyMean | T-total | 565 | 1566.258 | 1939.710 | -0.069 | 102.392 | 80.156 | -1.685 | 816.007 |
| L0-Basic | ElasticNet | G-perGPU | 565 | 336.914 | 588.959 | 0.901 | 14.647 | 17.953 | 20.070 | 112.272 |
| L0-Basic | ElasticNet | T-total | 565 | 345.178 | 584.479 | 0.903 | 15.808 | 20.559 | 8.036 | 128.036 |
| L0-Basic | ExtraTrees | G-perGPU | 565 | 290.845 | 499.360 | 0.929 | 13.248 | 16.590 | -13.198 | 100.565 |
| L0-Basic | ExtraTrees | T-total | 565 | 295.762 | 524.893 | 0.922 | 12.684 | 16.452 | -33.095 | 100.335 |
| L0-Basic | HistGradientBoosting | G-perGPU | 565 | 289.978 | 504.766 | 0.928 | 12.699 | 16.180 | -14.600 | 98.278 |
| L0-Basic | HistGradientBoosting | T-total | 565 | 366.573 | 641.772 | 0.883 | 14.264 | 19.043 | 78.212 | 126.682 |
| L0-Basic | RandomForest | G-perGPU | 565 | 296.227 | 507.219 | 0.927 | 12.566 | 16.528 | -11.142 | 99.756 |
| L0-Basic | RandomForest | T-total | 565 | 286.921 | 503.201 | 0.928 | 13.233 | 16.575 | 3.594 | 101.563 |
| L0-Basic | Ridge | G-perGPU | 565 | 345.712 | 600.987 | 0.897 | 14.763 | 18.251 | 22.824 | 113.814 |
| L0-Basic | Ridge | T-total | 565 | 359.989 | 597.069 | 0.899 | 15.853 | 21.758 | -0.227 | 136.100 |
| L0-Basic | XGBoost | G-perGPU | 565 | 291.602 | 489.160 | 0.932 | 12.843 | 15.951 | 11.065 | 96.600 |
| L0-Basic | XGBoost | T-total | 565 | 333.905 | 563.575 | 0.910 | 14.298 | 18.575 | 70.777 | 120.533 |
| L0A-ActiveAware | CatBoost | G-perGPU | 565 | 269.846 | 454.445 | 0.941 | 12.656 | 16.111 | -34.240 | 98.344 |
| L0A-ActiveAware | CatBoost | T-total | 565 | 295.236 | 547.274 | 0.915 | 12.814 | 16.740 | -72.936 | 102.897 |
| L0A-ActiveAware | DummyMean | G-perGPU | 565 | 510.852 | 822.627 | 0.808 | 22.463 | 23.702 | 9.240 | 147.822 |
| L0A-ActiveAware | DummyMean | T-total | 565 | 1566.258 | 1939.710 | -0.069 | 102.392 | 80.156 | -1.685 | 816.007 |
| L0A-ActiveAware | ElasticNet | G-perGPU | 565 | 331.465 | 542.793 | 0.916 | 13.769 | 17.420 | 29.578 | 107.973 |
| L0A-ActiveAware | ElasticNet | T-total | 565 | 349.504 | 593.602 | 0.900 | 14.563 | 19.335 | 32.662 | 123.161 |
| L0A-ActiveAware | ExtraTrees | G-perGPU | 565 | 283.163 | 476.023 | 0.936 | 12.668 | 16.568 | -6.671 | 100.553 |
| L0A-ActiveAware | ExtraTrees | T-total | 565 | 292.246 | 515.992 | 0.924 | 12.684 | 16.239 | -40.883 | 98.966 |
| L0A-ActiveAware | HistGradientBoosting | G-perGPU | 565 | 301.527 | 533.018 | 0.919 | 12.627 | 16.522 | 17.241 | 101.152 |
| L0A-ActiveAware | HistGradientBoosting | T-total | 565 | 348.156 | 600.503 | 0.898 | 14.409 | 18.917 | 72.049 | 125.082 |
| L0A-ActiveAware | RandomForest | G-perGPU | 565 | 274.305 | 458.762 | 0.940 | 12.188 | 16.399 | 5.187 | 99.610 |
| L0A-ActiveAware | RandomForest | T-total | 565 | 291.174 | 514.226 | 0.925 | 12.644 | 16.551 | -17.304 | 101.845 |
| L0A-ActiveAware | Ridge | G-perGPU | 565 | 337.661 | 549.464 | 0.914 | 14.193 | 17.651 | 27.713 | 109.445 |
| L0A-ActiveAware | Ridge | T-total | 565 | 356.634 | 601.741 | 0.897 | 14.661 | 19.722 | 36.692 | 127.190 |
| L0A-ActiveAware | XGBoost | G-perGPU | 565 | 273.385 | 466.722 | 0.938 | 12.355 | 15.845 | -1.045 | 96.526 |
| L0A-ActiveAware | XGBoost | T-total | 565 | 308.476 | 545.997 | 0.915 | 12.973 | 17.483 | 36.383 | 111.887 |
| L1-Deployment | CatBoost | G-perGPU | 565 | 225.192 | 371.173 | 0.961 | 10.545 | 14.142 | 16.156 | 85.147 |
| L1-Deployment | CatBoost | T-total | 565 | 299.723 | 573.843 | 0.906 | 11.926 | 15.494 | -48.458 | 97.283 |
| L1-Deployment | DummyMean | G-perGPU | 565 | 510.852 | 822.627 | 0.808 | 22.463 | 23.702 | 9.240 | 147.822 |
| L1-Deployment | DummyMean | T-total | 565 | 1566.258 | 1939.710 | -0.069 | 102.392 | 80.156 | -1.685 | 816.007 |
| L1-Deployment | ElasticNet | G-perGPU | 565 | 304.214 | 516.016 | 0.924 | 13.359 | 16.413 | 80.107 | 103.154 |
| L1-Deployment | ElasticNet | T-total | 565 | 321.657 | 547.195 | 0.915 | 14.025 | 18.570 | -42.785 | 114.126 |
| L1-Deployment | ExtraTrees | G-perGPU | 565 | 213.555 | 343.870 | 0.966 | 9.822 | 13.854 | 6.203 | 82.410 |
| L1-Deployment | ExtraTrees | T-total | 565 | 298.647 | 598.566 | 0.898 | 10.990 | 15.482 | -66.357 | 94.742 |
| L1-Deployment | HistGradientBoosting | G-perGPU | 565 | 210.284 | 350.111 | 0.965 | 9.508 | 13.982 | 16.470 | 83.494 |
| L1-Deployment | HistGradientBoosting | T-total | 565 | 303.156 | 536.205 | 0.918 | 12.328 | 16.506 | 18.227 | 107.512 |
| L1-Deployment | RandomForest | G-perGPU | 565 | 238.453 | 383.589 | 0.958 | 10.599 | 14.679 | 22.312 | 88.568 |
| L1-Deployment | RandomForest | T-total | 565 | 285.336 | 535.484 | 0.919 | 11.511 | 15.464 | -33.474 | 94.561 |
| L1-Deployment | Ridge | G-perGPU | 565 | 371.255 | 787.967 | 0.824 | 14.094 | 17.536 | 142.370 | 114.940 |
| L1-Deployment | Ridge | T-total | 565 | 331.804 | 557.156 | 0.912 | 15.525 | 19.711 | -54.411 | 120.928 |
| L1-Deployment | XGBoost | G-perGPU | 565 | 202.698 | 322.671 | 0.970 | 9.467 | 13.773 | 21.527 | 82.501 |
| L1-Deployment | XGBoost | T-total | 565 | 281.165 | 541.243 | 0.917 | 12.237 | 16.979 | -4.297 | 112.634 |
| L2-StaticCore | CatBoost | G-perGPU | 448 | 197.228 | 326.254 | 0.969 | 10.107 | 14.521 | -54.824 | 89.488 |
| L2-StaticCore | CatBoost | T-total | 448 | 241.087 | 467.634 | 0.936 | 11.017 | 15.297 | -54.837 | 96.468 |
| L2-StaticCore | DummyMean | G-perGPU | 448 | 394.243 | 644.209 | 0.878 | 20.173 | 21.709 | -27.648 | 135.794 |
| L2-StaticCore | DummyMean | T-total | 448 | 1564.876 | 1905.455 | -0.069 | 117.637 | 85.700 | 3.988 | 820.146 |
| L2-StaticCore | ElasticNet | G-perGPU | 448 | 286.675 | 470.914 | 0.935 | 15.131 | 17.308 | 86.297 | 109.098 |
| L2-StaticCore | ElasticNet | T-total | 448 | 314.216 | 654.815 | 0.874 | 13.340 | 18.887 | -121.381 | 114.740 |
| L2-StaticCore | ExtraTrees | G-perGPU | 448 | 187.138 | 295.335 | 0.974 | 9.717 | 13.325 | -24.002 | 82.421 |
| L2-StaticCore | ExtraTrees | T-total | 448 | 232.589 | 459.624 | 0.938 | 10.386 | 14.503 | -84.555 | 90.276 |
| L2-StaticCore | HistGradientBoosting | G-perGPU | 448 | 162.991 | 241.744 | 0.983 | 9.569 | 13.455 | -14.461 | 82.193 |
| L2-StaticCore | HistGradientBoosting | T-total | 448 | 205.311 | 391.109 | 0.955 | 9.463 | 14.460 | -69.116 | 88.021 |
| L2-StaticCore | RandomForest | G-perGPU | 448 | 187.258 | 277.510 | 0.977 | 9.723 | 14.121 | -21.866 | 87.378 |
| L2-StaticCore | RandomForest | T-total | 448 | 216.445 | 418.825 | 0.948 | 10.319 | 14.018 | -37.249 | 87.534 |
| L2-StaticCore | Ridge | G-perGPU | 448 | 339.967 | 644.274 | 0.878 | 15.479 | 18.051 | 140.039 | 116.975 |
| L2-StaticCore | Ridge | T-total | 448 | 306.336 | 610.893 | 0.890 | 14.207 | 18.481 | -98.402 | 114.283 |
| L2-StaticCore | XGBoost | G-perGPU | 448 | 179.253 | 273.815 | 0.978 | 8.981 | 13.771 | -10.416 | 84.901 |
| L2-StaticCore | XGBoost | T-total | 448 | 205.961 | 380.398 | 0.957 | 9.665 | 14.744 | -77.072 | 89.601 |
| L3-Workload | CatBoost | G-perGPU | 448 | 173.421 | 286.807 | 0.976 | 9.065 | 11.096 | -25.780 | 71.168 |
| L3-Workload | CatBoost | T-total | 448 | 241.877 | 510.831 | 0.923 | 9.457 | 12.700 | -43.329 | 81.632 |
| L3-Workload | DummyMean | G-perGPU | 448 | 394.243 | 644.209 | 0.878 | 20.173 | 21.709 | -27.648 | 135.794 |
| L3-Workload | DummyMean | T-total | 448 | 1564.876 | 1905.455 | -0.069 | 117.637 | 85.700 | 3.988 | 820.146 |
| L3-Workload | ElasticNet | G-perGPU | 448 | 304.397 | 509.176 | 0.924 | 15.846 | 17.125 | 114.403 | 107.550 |
| L3-Workload | ElasticNet | T-total | 448 | 296.477 | 504.694 | 0.925 | 14.524 | 19.830 | -17.354 | 132.275 |
| L3-Workload | ExtraTrees | G-perGPU | 448 | 195.894 | 379.511 | 0.958 | 8.427 | 11.801 | 41.935 | 73.295 |
| L3-Workload | ExtraTrees | T-total | 448 | 211.871 | 430.732 | 0.945 | 8.603 | 12.128 | -73.149 | 76.121 |
| L3-Workload | HistGradientBoosting | G-perGPU | 448 | 189.152 | 319.538 | 0.970 | 9.111 | 11.755 | 3.838 | 74.412 |
| L3-Workload | HistGradientBoosting | T-total | 448 | 195.880 | 422.938 | 0.947 | 7.143 | 11.651 | -87.650 | 70.941 |
| L3-Workload | RandomForest | G-perGPU | 448 | 184.907 | 285.819 | 0.976 | 9.766 | 12.401 | -15.362 | 77.095 |
| L3-Workload | RandomForest | T-total | 448 | 199.793 | 428.036 | 0.946 | 8.748 | 11.983 | -33.937 | 75.109 |
| L3-Workload | Ridge | G-perGPU | 448 | 279.500 | 439.350 | 0.943 | 15.109 | 16.832 | 93.265 | 104.054 |
| L3-Workload | Ridge | T-total | 448 | 321.492 | 601.040 | 0.894 | 14.476 | 19.888 | -30.577 | 133.956 |
| L3-Workload | XGBoost | G-perGPU | 448 | 172.646 | 285.310 | 0.976 | 8.336 | 11.410 | 7.798 | 72.776 |
| L3-Workload | XGBoost | T-total | 448 | 177.683 | 319.855 | 0.970 | 8.098 | 11.060 | -53.484 | 68.175 |
| L4-CompactPhysics | CatBoost | G-perGPU | 305 | 228.771 | 365.377 | 0.961 | 10.168 | 11.752 | 20.990 | 75.151 |
| L4-CompactPhysics | CatBoost | T-total | 305 | 290.472 | 542.915 | 0.915 | 9.932 | 13.160 | -61.738 | 86.860 |
| L4-CompactPhysics | DummyMean | G-perGPU | 305 | 483.710 | 740.614 | 0.842 | 23.334 | 24.460 | -8.243 | 155.088 |
| L4-CompactPhysics | DummyMean | T-total | 305 | 1672.056 | 2004.595 | -0.159 | 115.706 | 84.736 | -2.308 | 865.559 |
| L4-CompactPhysics | ElasticNet | G-perGPU | 305 | 349.146 | 575.502 | 0.904 | 14.900 | 16.443 | 116.648 | 107.088 |
| L4-CompactPhysics | ElasticNet | T-total | 305 | 239.413 | 377.210 | 0.959 | 11.064 | 14.480 | -62.224 | 89.320 |
| L4-CompactPhysics | ExtraTrees | G-perGPU | 305 | 284.366 | 524.291 | 0.921 | 9.998 | 13.631 | -37.750 | 84.119 |
| L4-CompactPhysics | ExtraTrees | T-total | 305 | 226.913 | 470.230 | 0.936 | 8.327 | 11.583 | -40.561 | 73.431 |
| L4-CompactPhysics | HistGradientBoosting | G-perGPU | 305 | 251.785 | 439.682 | 0.944 | 9.696 | 12.916 | 56.240 | 82.052 |
| L4-CompactPhysics | HistGradientBoosting | T-total | 305 | 190.427 | 385.780 | 0.957 | 7.208 | 10.491 | -8.246 | 66.536 |
| L4-CompactPhysics | RandomForest | G-perGPU | 305 | 277.481 | 521.424 | 0.922 | 10.510 | 12.515 | 76.198 | 80.830 |
| L4-CompactPhysics | RandomForest | T-total | 305 | 223.859 | 445.900 | 0.943 | 8.399 | 10.999 | -65.224 | 69.709 |
| L4-CompactPhysics | Ridge | G-perGPU | 305 | 335.416 | 630.337 | 0.885 | 13.998 | 15.977 | 171.976 | 106.434 |
| L4-CompactPhysics | Ridge | T-total | 305 | 317.643 | 585.768 | 0.901 | 12.928 | 18.242 | -87.717 | 117.961 |
| L4-CompactPhysics | XGBoost | G-perGPU | 305 | 183.111 | 293.201 | 0.975 | 8.184 | 10.676 | 7.430 | 67.307 |
| L4-CompactPhysics | XGBoost | T-total | 305 | 185.233 | 362.211 | 0.962 | 6.843 | 9.892 | -83.193 | 62.478 |

## RQ8：分层development candidates及适用边界

L0/L1/L2（并保留L3）候选由各层main nested OOF选择算法和target，在完整对应support上独立3-fold GroupCV从冻结候选列表选最终参数并拟合；metadata设置development_candidate_not_external_validated=true。它们是研发候选，不是外部验证后的通用GPU或node模型。选择规则先锁定，再运行post-hoc stresses。

| feature_level | algorithm | target_strategy | cohort | n_rows | mae | per_gpu_mae |
|---|---|---|---|---|---|---|
| L0-Basic | CatBoost | T-total | full | 565 | 283.519 | 98.112 |
| L1-Deployment | XGBoost | G-perGPU | full | 565 | 202.698 | 82.501 |
| L2-StaticCore | HistGradientBoosting | G-perGPU | static | 448 | 162.991 | 82.193 |
| L3-Workload | XGBoost | G-perGPU | static | 448 | 172.646 | 72.776 |

## 泛化压力与校准（post-hoc only）

以下为L0/L1/L2锁定研发候选在leave-one-lineage、within-task-lineage、architecture holdout上的固定参数压力测试，完全使用保存manifest；不回流选择、超参数或feature tiers。最困难的slice如下，全部结果见generalization_stress.csv：

| feature_level | scheme | held_out_group | task | n_rows | n_models | mae | per_gpu_mae | signed_bias |
|---|---|---|---|---|---|---|---|---|
| L0-Basic | parent_lineage | DeepSeek | nan | 22 | 2 | 2091.583 | 261.448 | 2091.583 |
| L0-Basic | within_task_lineage | DeepSeek | gpqa | 11 | 2 | 1587.677 | 198.460 | -1587.677 |
| L0-Basic | within_task_lineage | DeepSeek | lm-arena-chat | 11 | 1 | 1252.804 | 156.600 | -1227.376 |
| L1-Deployment | parent_lineage | DeepSeek | nan | 22 | 2 | 1201.481 | 150.185 | 1175.860 |
| L2-StaticCore | parent_lineage | DeepSeek | nan | 22 | 2 | 1081.028 | 135.129 | 1060.305 |
| L0-Basic | within_task_lineage | Llama | lm-arena-chat | 117 | 7 | 965.200 | 205.230 | -714.419 |
| L2-StaticCore | within_task_lineage | DeepSeek | gpqa | 11 | 2 | 799.191 | 99.899 | 185.632 |
| L0-Basic | parent_lineage | Llama | nan | 117 | 7 | 713.835 | 153.822 | -301.464 |
| L1-Deployment | within_task_lineage | DeepSeek | gpqa | 11 | 2 | 705.753 | 88.219 | 516.546 |
| L0-Basic | within_task_lineage | Qwen | gpqa | 83 | 6 | 604.289 | 227.963 | 187.866 |
| L0-Basic | architecture | mixture_of_experts | nan | 306 | 15 | 571.617 | 126.116 | 268.032 |
| L1-Deployment | architecture | mixture_of_experts | nan | 306 | 15 | 534.556 | 114.093 | 464.417 |

Task-lineage confounding、GPU composition、task composition单独记录。只有H100/B200两个reference points，不能声明跨GPU世代；hybrid支持主要为Nemotron，架构holdout也与谱系混杂。Overall/8GPU/top10校准图及CSV用于判断压缩、异方差和系统偏差，不用于追加调参。

## 敏感性分支的配对证据

L1→tokenizer-free L1T：在相同支持集（565行）上，16组 algorithm×target 比较的ΔMAE范围为 0.0 至 55.5 W；0组95%模型簇区间全部低于0，1组全部高于0，其余跨0。完整明细包括row-weighted和equal-model口径，见 feature_increment_effects.csv。

L2→架构扩展L2X：在相同支持集（448行）上，16组 algorithm×target 比较的ΔMAE范围为 -17.2 至 126.3 W；0组95%模型簇区间全部低于0，1组全部高于0，其余跨0。完整明细包括row-weighted和equal-model口径，见 feature_increment_effects.csv。

L4→旧B5：在相同支持集（305行）上，16组 algorithm×target 比较的ΔMAE范围为 -17.5 至 137.9 W；0组95%模型簇区间全部低于0，1组全部高于0，其余跨0。完整明细包括row-weighted和equal-model口径，见 feature_increment_effects.csv。

L1→task label：在相同支持集（565行）上，16组 algorithm×target 比较的ΔMAE范围为 -75.1 至 48.9 W；0组95%模型簇区间全部低于0，1组全部高于0，其余跨0。完整明细包括row-weighted和equal-model口径，见 feature_increment_effects.csv。

这些分支不用于改选主研发候选。Tokenizer-free统计在本benchmark中仅有三组任务级长度分布，不能据此宣称跨新workload泛化。

## 独立审查与透明语义修正

审查发现L2X把明确禁用滑动窗口的276条Qwen记录/12模型误当unknown；原config原字节哈希均与冻结Step3 architecture evidence一致，历史文件未改。保留原protocol/master和superseded evidence；在冻结`l2x_semantic_amendment.json`后仅重跑16个L2X单元，原448行、folds、候选、objective不变，208个原始单元保留。`phase5_master_table.parquet`是原预修正快照；L2X正式有效输入为`phase5_l2x_amended_table.parquet`，禁用window=0、applicable=0、unknown=0。上述敏感性证据采用修正后的OOF；候选锁保持原样，source_matrix哈希对应归档原矩阵，而全部main行数值保持不变。

OOF原始/修正指纹已由独立验证器重建，stress cache加入共享建模代码和protocol来源；错误指纹的fault-injection回归必须失败。完整审查发现、修复和审查限制见`final_code_review.md`与`source_integrity_issue.md`。修正是语义缺陷修复，不是看结果后追加特征或扩搜索。

## 数据、口径和可复现性

| feature_level | n_rows | n_models | n_lineages | n_architectures |
|---|---|---|---|---|
| L0-Basic | 565 | 27 | 6 | 3 |
| L0A-ActiveAware | 565 | 27 | 6 | 3 |
| L1-Deployment | 565 | 27 | 6 | 3 |
| L2-StaticCore | 448 | 20 | 5 | 3 |
| L3-Workload | 448 | 20 | 5 | 3 |
| L4-CompactPhysics | 305 | 14 | 2 | 2 |
| L3E-Effective | 383 | 18 | 4 | 3 |
| L2X-StaticExtended | 448 | 20 | 5 | 3 |
| L4B5-OldB5Sensitivity | 305 | 14 | 2 | 2 |
| L1-TaskSensitivity | 565 | 27 | 6 | 3 |
| L1T-TokenizerFreeTextProxy | 565 | 27 | 6 | 3 |

正式X仅执行前可知；model ID/lineage/subfamily只用于分组/诊断，x_protocol_task仅最后的task sensitivity。所有预处理fold-local，类别未知ignore；真正unknown不能当N/A。Source/protocol/candidate/fold/code/version hashes和checkpoint指纹可复核；ElasticNet弱正则相关输入的数值收敛通过同objective ridge初始化修复，候选空间未改，剩余失败逐项记录，不删失败fold后宣称完整。

## 限制和下一阶段建议

1. 本阶段统一8候选而非默认24，探索预算较小；不可在已看outer结果后偷偷扩搜索并保留原“冻结”声明。
2. Best-of-level最低OOF有winner-selection bias；Phase4 Confirmatory已看，本阶段不产生新的blind confirmatory。
3. N/A编码与理想weight residency/compute proxy是有边界的结构表示；不对应实测utilization、bandwidth或per-card功率。
4. 明确分层L0/L1/L2信息要求，可供后续真实事前预测接口使用；新增外部数据再验证困难谱系、高功率和GPU-generation外推。
5. 相邻信息收益和target收益若CI跨0，结论为当前模型簇证据不足；不能将“未显著”写成物理机制不存在。区间是当前OOF管线下的模型簇重采样区间，多个比较未作family-wise校正，不覆盖跨数据集迁移或全部模型选择不确定性。
6. H100/B200实际SKU未获得逐run确认，硬件额定功率不是配置power cap。继承的H100 MXFP4算力lookup使用BF16参考代理，不代表H100具有原生FP4峰值；不能把该字段当实测或严格precision-matched能力。

六类最低图表均有机器可读来源，notebook只加载persisted evidence并绘图；完整验收见verification_receipt.json。所有step1–step6正式文件保持只读。
