# Phase 1.7：完整事前特征补齐与建模输入表定稿

本目录承接 `step1/` 与 `step2/`，范围严格限制为 ML.ENERGY V3 的 694 条 text-LLM inference runs。本阶段补齐 planned/effective workload、canonical 模型结构、physics-derived 事前 proxy，并形成一行一个 run 的 master feature table；不训练正式预测模型。

## 安全边界

- 只使用预测执行前已知或可确定计算的输入特征。
- GPQA 和 Google Gemma 使用用户已获批的合法访问；Meta Llama 访问被拒，不通过镜像或替代 tokenizer 绕过。
- 只下载 config/tokenizer/chat-template 等静态文件，禁止模型权重、完整 timeline 和 Prometheus。
- 凭证通过终端隐藏输入，`step3/data/`、`step3/.venv/` 和 `step3/.secrets/` 均不进入 Git。

## 本地环境

```powershell
python -m venv step3/.venv
step3/.venv/Scripts/python.exe -m pip install -r step3/requirements-phase17.txt
step3/.venv/Scripts/python.exe -m unittest discover -s step3/tests -v
```

完整执行顺序和验收点见 `docs/superpowers/plans/2026-09-18-phase1-7-feature-table.md`。

## 最终产物

- `analysis/master_feature_table.parquet` / `.csv`：694 行一行一个 run 的建模输入主表。
- `analysis/feature_role_manifest.json`：每列的 X/Y/诊断/溯源角色与事前可用标记。
- `docs/master_feature_dictionary.md`：字段字典和 eligibility 定义。
- `reports/eligibility_coverage.json`：全体/稳定以及 task、model family、GPU、architecture family 分层覆盖。
- `analysis/rapl_zone_identity_evidence.json` 和 `reports/rapl_zone_identity_check.md`：RAPL zone 1 限定证据检索。
- `notebooks/phase1_7_feature_completion_audit.ipynb`：已从头到尾执行的定量审计 Notebook。
- `reports/phase1_7_feature_completion_report.md`：最终发现、限制与 Phase 2 建议。
- `reports/verification_receipt.json`：最终自动验收回执。

## 重现最终审计

```powershell
step3/.venv/Scripts/python.exe -m step3.scripts.build_master_table
step3/.venv/Scripts/python.exe -m step3.scripts.audit_rapl_identity
step3/.venv/Scripts/python.exe -m step3.scripts.build_notebook
step3/.venv/Scripts/python.exe -m step3.scripts.validate_phase17
step3/.venv/Scripts/python.exe -m unittest step3.tests.test_phase17 -v
```

`step3/data/`、`step3/.venv/` 和 Notebook HTML 是可重建的忽略产物，不需上传 GitHub。建模必须以 `feature_role_manifest.json` 为泄漏边界，并按 eligibility cohort 分别报告结果。
