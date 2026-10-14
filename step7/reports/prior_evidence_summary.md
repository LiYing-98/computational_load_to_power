# Phase 5 继承的既有证据

此文件在 Phase5 建模实现和训练之前生成。数据对象为 ML.ENERGY V3 GPU 侧稳定、高负载稳态、文本 LLM inference。

| 继承事实 | 冻结证据文件 | Phase5 含义 |
|---|---|---|
| Phase4 565 stable runs、27 models、117 Llama；P0/P1/P2 全覆盖 | step6/analysis/phase4_power_modeling_table.parquet；step6/analysis/coverage_by_lineage.csv | L0/L0A/L1 不因 detailed-static 缺失缩减 |
| RandomForest + P2 + L1-loss 候选；development OOF MAE 272.4 W | step6/config/locked_candidate_phase4.json；step6/reports/main_model_comparison.csv | 作为历史基线，不硬编码 Phase5 冠军 |
| 116行 Confirmatory MAE 295.3 W 已被查看 | step6/analysis/confirmatory_evaluation_receipt.json；step6/reports/confirmatory_metrics.csv | 不可再次称为 blind confirmatory |
| 8卡 total MAE 720.0 W，平均单 GPU 误差为 90.0 W/GPU；1卡 total/per-GPU MAE 127.7 W | step6/reports/error_slices.csv | 同时报告 total/per-GPU，不能只用 total 误差推断8卡更难 |
| 448行 static subset MAE 189.7/176.9/183.0 W（无硬件/aggregate/full硬件） | step6/reports/static_subset_comparison.csv；step6/scripts/analyze_phase4.py | 更小支持集的绝对误差不能充当详细结构增益 |
| Benchmark input intensity 有支持性增益；effective 额外收益不稳定 | step4/reports/ablation_summary.csv；step4/reports/phase2_0_baseline_report.md | L3优先benchmark，L3E用相同支持集 sensitivity |
| Physics增益依赖学习器，B5归一化比率多数无益 | step5/reports/feature_ablation.csv；step5/reports/hardware_label_sensitivity.csv；step5/reports/phase3_enhanced_modeling_report.md | L4精简；旧B5仅sensitivity |
| H100/B200仅两个reference proxy，SKU仍ambiguous；H100无原生FP4峰值 | step5/config/gpu_hardware_specs.json；step5/config/hardware_source_manifest.json；step6/reports/phase4_power_model_report.md | 不作跨GPU世代结论；显式标记匹配compute fallback |
| Actual batch、throughput、latency、duration、utilization是事后量 | step3/analysis/feature_role_manifest.json；step6/config/feature_bundles_phase4.json | 永久禁止进入正式X |

历史文件保持只读；Phase5 的每个数字和增益由本阶段持久化结果重新计算。已有 benchmark 数值和有效长度的定义、冻结模板与scope限制见 step3/analysis/planned_workload_provenance.json 和 step3/analysis/effective_workload_provenance.json。
