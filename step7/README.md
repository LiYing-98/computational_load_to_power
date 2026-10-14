# ML.ENERGY Phase 5.0 — Step 7

依据 `Codex_Phase5_Taskbook.docx`（SHA-256 `71573fb954ba7d90ff62bfcdab48662922b7063082d568adbf288694239a8f56`）开展 L0–L4 事前信息层级 × total/per-GPU target 研究。

研究口径：GPU 侧、stable、文本 LLM inference、high-load / near-saturation steady-state。Step1–6 为只读冻结输入；Phase4 的既有 Confirmatory 已被查看，本阶段不作新的盲测声明。

进度与接续：先读 `reports/progress.md` 和 `docs/2026-10-13-phase5-implementation-plan.md`，然后验证 source/split/candidate/checkpoint 指纹，从未完成单元接续。

运行时沿用 `step4/.venv/Scripts/python.exe` 和已冻结的 `step5/.packages` 后端。命令从仓库根目录执行。正式脚本、结果、报告、notebook 与测试均在本目录；临时 checkpoint 忽略。

Phase5已完成并停止研究（2026-10-14）。当前线程自动化`phase5-step7`已暂停；原配置为2026-10-13 21:00首次、此后2小时。保留配置和完整接续记录，不再自动重跑。

最终验收：224/224当前有效单元、20/20 Step7测试、9/9独立验证、272配对比较/544000模型簇draws、6640误差切片、4研发候选、52post-hoc压力单元、6类图和执行notebook/HTML。一次独立全分支审查的3项重要发现已修复并经回归/全suite验证；历史`needs_review`状态已关闭。主候选未外部验证，不能把best-of-level OOF当无偏最终性能。

## 运行与复核

从仓库根目录使用已有Phase2环境（依赖参考`step4/requirements-phase20.txt`及`step5/requirements-phase30.txt`），命令为：

```powershell
# 仅查看与验证；不会启动新训练
& 'step4/.venv/Scripts/python.exe' -m step7.scripts.validate_phase50 --progress-only

# 确认不存在当前矩阵或resume驱动后，才可按断点接续
& 'step4/.venv/Scripts/python.exe' -m step7.scripts.resume_phase5 --workers 3
```

真实最终状态见`analysis/pipeline_state.json`、`analysis/matrix_progress.json`和`reports/verification_receipt.json`。通常审阅无需再启动resume驱动；如需复核测试，应按 tests→audit_feature_dictionary→validate_phase50顺序恢复修正后的审计说明。忽略的`models/checkpoints/`是本机接续缓存，不能拿部分缓存当完整研究证据。逐字段公式/单位/适用性字典见`analysis/feature_dictionary_phase5.csv`；原始tokenizer-free请求内容不在本阶段重复发布。无延期审查问题；L2X线性基线负值风险已单独诊断，没有clip/删数据/调参。

独立审查后的L2X语义修正见`reports/source_integrity_issue.md`、`reports/final_code_review.md`和`config/l2x_semantic_amendment.json`。原master/protocol保持冻结；L2X有效输入为`analysis/phase5_l2x_amended_table.parquet`，只重跑16敏感性单元，208原单元保留。当前矩阵必须由amendment-aware驱动接续，**不要运行旧`run_matrix --collect-only`或原始全矩阵runner覆盖修正后的L2X**。候选锁仍指向归档原矩阵，主结果不受修正影响。
