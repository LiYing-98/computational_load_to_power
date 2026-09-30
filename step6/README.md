# ML.ENERGY Phase 4.0

本目录是 ML.ENERGY V3 **GPU 侧高负载稳态文本 LLM inference** 的功率主模型定型研究。正式目标只有聚合 GPU 平均功率 `y_gpu_avg_power_watts`。

## 优先阅读

- 科学报告：`reports/phase4_power_model_report.md`
- 执行后证据 notebook：`notebooks/phase4_power_modeling.ipynb`（同时提供 HTML）
- 一次性 Confirmatory 指标：`reports/confirmatory_metrics.csv`
- 开发集模型比较：`reports/main_model_comparison.csv`
- 逐行 Confirmatory 预测：`analysis/confirmatory_predictions.parquet`
- 完整 565 条诊断 OOF：`analysis/full_cohort_oof_predictions.parquet`
- 独立验收回执：`reports/verification_receipt.json`

## 主要结论

锁定候选为 `RandomForest + P2-HardwareStatic + L1`。development model-group OOF MAE 为 272.4 W；候选锁定后的一次性 116 行 Confirmatory MAE 为 295.3 W。P0/P1/P2 覆盖全部 565 条 stable runs，包含 117 条 Llama。跨 parent lineage 和 8-GPU/top-power 外推仍是主要限制。

## 重现

从仓库根目录运行：

```powershell
& 'step4\.venv\Scripts\python.exe' -m step6.scripts.build_modeling_table
& 'step4\.venv\Scripts\python.exe' -m step6.scripts.build_splits
& 'step4\.venv\Scripts\python.exe' -m step6.scripts.run_development_matrix
& 'step4\.venv\Scripts\python.exe' -m step6.scripts.analyze_development
& 'step4\.venv\Scripts\python.exe' -m step6.scripts.run_loss_strategies
& 'step4\.venv\Scripts\python.exe' -m step6.scripts.evaluate_locked_candidate
& 'step4\.venv\Scripts\python.exe' -m step6.scripts.analyze_phase4
& 'step4\.venv\Scripts\python.exe' -m step6.scripts.build_notebook
& 'step4\.venv\Scripts\python.exe' -m step6.scripts.validate_phase40
& 'step4\.venv\Scripts\python.exe' -m unittest discover -s step6/tests -v
```

模型训练支持忽略的 checkpoint；Confirmatory 评价是幂等的，已有结果只校验哈希，不会重拟合或改写。独立验证器会校验实际任务书 SHA-256。若任务书不在默认位置，可通过 `MLENERGY_PHASE4_TASKBOOK` 指向同一文件。

## 科学边界

- 正式 X 仅含执行前可知信息；model ID、lineage、subfamily、任务结果及 post-run telemetry 均不进入模型。
- P2 硬件规格是 Phase 3 冻结的 reference proxy；H100/B200 两点不足以证明跨 GPU 世代泛化。
- Lineage、within-task lineage 和 architecture holdout 使用单独的完整 565 条 post-lock diagnostic manifests，包含原 Confirmatory 模型，但从不参与候选选择或改写一次性 Confirmatory 指标。
- 本目录不包含 energy/token、CPU/DRAM/node、Training、Diffusion、瞬态或 arrival-rate 建模。
- 不需要上传 `models/checkpoints/` 或 Python 缓存；它们已被忽略。正式审查应上传 `step6/` 中受 Git 跟踪的配置、脚本、测试、表、模型、图、报告和 notebook。
