# 修正后负功率预测诊断（post-hoc，不回流调参）

范围：最终224单元的保存OOF。主层96个单元负预测数为0；全部G-perGPU单元负预测数为0。58个负预测均来自L2X sensitivity的T-total线性基线：Ridge29、ElasticNet29，对应同一批29个unique run、2个Gemma模型、outer fold0。不是58个不同run，也不是实测功率为负。

| 学习器 | held-out model | 负预测数 | 最小预测W | 最大负预测W | 最小实测W |
|---|---|---:|---:|---:|---:|
| Ridge | google/gemma-3-12b-it | 13 | -1058.683 | -825.446 | 424.237 |
| Ridge | google/gemma-3-27b-it | 16 | -892.288 | -525.698 | 510.421 |
| ElasticNet | google/gemma-3-12b-it | 13 | -1225.867 | -1003.679 | 424.237 |
| ElasticNet | google/gemma-3-27b-it | 16 | -1096.630 | -704.005 | 510.421 |

证据：`reports/main_model_matrix.csv`的negative_predictions及`analysis/oof_predictions_L2X-StaticExtended__static__Ridge__T-total.parquet`、ElasticNet同名OOF。保存selection_metadata_json显示fold0选择Ridge alpha=74.05684692262442；ElasticNet alpha=0.4489251258218603、l1_ratio=0.8。参数来自各自inner CV，未改动。

诊断：两者都是无非负约束的线性回归，不约束输出物理范围。异常集中在同一held-out Gemma结构组合，有限模型组下扩展静态字段的线性外推不稳定；所有预测仍为有限数、完整group OOF，不是跳过失败fold或丢失恢复卡数。Gemma窗口为1024；修正后Qwen明确禁用编码0，未知仍为NaN，完整训练字段与fold可从冻结/修正表及manifest检查。未做逐特征因果分解，因此不声称是某一个字段单独造成负值。

处理：原值、误差、偏差及bootstrap全部保留，没有clip、删除run或重调参。它们不是本阶段development candidates；主候选和原锁不变。当前结论是L2X线性扩展有物理可用性风险，不部署这些无约束敏感性基线。下一阶段如研究正值目标变换/非负约束，应另行预注册并用新证据验证，不能回头改本阶段已查看的结果。
