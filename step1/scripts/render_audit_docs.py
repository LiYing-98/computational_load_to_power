#!/usr/bin/env python3
"""Render Phase 1.5 Markdown deliverables from reviewed JSON audit outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("step1")


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "analysis" / name).read_text(encoding="utf-8"))


def md_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", " ")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def render_field_dictionary(full: dict[str, Any]) -> None:
    lines = [
        "# `runs/llm.parquet` 完整字段字典（文本 LLM inference 子集）",
        "",
        f"审计粒度：{full['grain']}。原表 {full['source_population']['parquet_all_llm_and_mllm_rows']} 行；"
        f"严格筛选 `gpqa`、`lm-arena-chat`、`sourcegraph-fim` 后为 {full['row_count']} 行、"
        f"{full['column_count']} 列。空值按 Arrow null 统计，空字符串另列。",
        "",
        "| 字段 | Arrow 类型 | 分类 | 可进入 `X_pre` | null | 空字符串 | 审计解释 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for field in full["columns"]:
        item = full["field_classification"][field]
        null_count = full["nulls"][field]["count"]
        blank_count = full.get("blank_strings", {}).get(field, {}).get("count", 0)
        lines.append(
            f"| `{field}` | `{full['field_types'].get(field, 'unknown')}` | "
            f"`{item['category']}` | {'是' if item['allow_as_x_pre'] else '否'} | "
            f"{null_count} | {blank_count} | {md_value(item['rationale'])} |"
        )
    lines.extend(
        [
            "",
            "## 键与重复解释",
            "",
            f"- 37 列整行完全重复：{full['exact_duplicates']['duplicate_groups']} 组。",
            f"- 第一阶段可见的 8 字段键出现 {full['keys']['phase1_visible']['duplicate_groups']} 个重复组、"
            f"涉及 {full['keys']['phase1_visible']['rows_in_duplicate_groups']} 行。",
            f"- 加入 `seed` 与 `num_request_repeats` 后，{full['keys']['repeat_aware']['unique_keys']} 行全部唯一。"
            "因此此前的“重复”是工作量重复次数不同，不是无意义重复记录。",
            "- `seed` 在 694 行中恒为 48105，数据不能直接估计跨随机种子的测量波动。",
            "",
            "## 口径",
            "",
            "四个恒等式校验全部通过：`energy / duration = avg_power`、"
            "`energy_per_token × output_throughput = avg_power`、"
            "`energy_per_token × avg_output_len = energy_per_request`、"
            "`total_output_tokens / completed_requests = avg_output_len`。"
            "功率与能量均为稳态窗口内所有 GPU 的聚合 GPU 侧口径，不代表 CPU、整机或数据中心功耗。",
        ]
    )
    write(ROOT / "docs" / "full_field_dictionary.md", "\n".join(lines))


def render_workload_matrix(workload: dict[str, Any]) -> None:
    lines = [
        "# 三类文本 LLM inference 工作量字段矩阵",
        "",
        f"官方 benchmark 源码固定在提交 `{workload['benchmark_code']['commit']}`；"
        f"代表性 metadata 共 {workload['scope']['selected_run_count']} 份，保留 timeline payload 为 0。",
        "",
        "| 任务 | 官方数据集 | split/subset | 唯一请求数 | 代表样本重复次数 | 单请求输出上限 | endpoint | 输入 token 事前状态 |",
        "|---|---|---|---:|---|---:|---|---|",
    ]
    for task, spec in workload["task_matrix"].items():
        split = spec["dataset_split"]
        if spec.get("dataset_subset"):
            split += f" / {spec['dataset_subset']}"
        lines.append(
            f"| `{task}` | `{spec['dataset']}` | `{split}` | {spec['num_unique_prompts']} | "
            f"{md_value(spec['observed_repeat_counts'])} | {spec['max_output_tokens']} | "
            f"`{spec['endpoint_type']}` | 未预存；仅在冻结请求、tokenizer 与模板后可事前推导 |"
        )
    lines.extend(
        [
            "",
            "统一流量配置：代表样本中 `request_rate=inf`、`burstiness=1.0`、"
            "`max_concurrency=null`、`max_num_batched_tokens=null`。源码默认采样参数为 "
            "`top_p=0.95`、`temperature=0.8`、`ignore_eos=false`；这些参数未写入所审计的结果头部，"
            "应在下一阶段配置清单中显式固化。",
            "",
            "## 代表性结果头部字段",
            "",
            "| JSON 路径 | 分类 | 可进入 `X_pre` | 出现数 | 任务覆盖 | 示例 |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for path, item in workload["field_audit"].items():
        lines.append(
            f"| `{path}` | `{item['category']}` | {'是' if item['allow_as_x_pre'] else '否'} | "
            f"{item['present']} | {', '.join(item['availability_by_task'])} | {md_value(item['examples'][:2])} |"
        )
    lines.extend(
        [
            "",
            "## 关键解释",
            "",
            "- `num_prompts` 可在执行前由 `num_unique_prompts × num_request_repeats` 得到。",
            "- 结果文件里的 `total_input_tokens` 虽可理论上提前重构，但该列本身是执行后记录；"
            "建模时必须使用独立的事前重构值，不能直接回填结果列。",
            "- `max_output_tokens` 只是上限，不是实际或期望输出长度；`total_output_tokens`、吞吐、时长、"
            "完成数、ITL、平均 batch 均是执行后泄漏变量。",
            "- Sourcegraph FIM 的 prompt 格式随模型 tokenizer 类型变化；LM Arena 多轮拼接、GPQA 选项随机化"
            "也受固定 seed 与实现版本约束，所以必须冻结源码、数据集 revision、请求清单和 tokenizer revision。",
        ]
    )
    write(ROOT / "docs" / "workload_field_matrix.md", "\n".join(lines))


def render_x_pre(workload: dict[str, Any]) -> None:
    xpre = workload["x_pre"]
    lines = [
        "# `X_pre` 特征目录与泄漏边界",
        "",
        "原则：预测时只能使用任务发起前已经给定，或能从冻结配置与输入确定性计算出的信息。"
        "任何来自实际生成、调度、吞吐、时长或能源测量的量均不得进入输入。",
        "",
        "## 可直接使用的事前字段",
        "",
    ]
    lines.extend(f"- `{field}`" for field in xpre["raw_allowed"])
    lines.extend(["", "## 可事前推导的字段", ""])
    lines.extend(f"- `{field}`" for field in xpre["pre_derived_allowed"])
    lines.extend(["", "## 禁止作为输入的执行后字段", ""])
    lines.extend(f"- `{field}`" for field in xpre["forbidden_post_run"])
    lines.extend(
        [
            "",
            "## 标识符风险",
            "",
            "`model_id`/`nickname` 虽在执行前已知，但直接编码会让模型记忆具体模型身份。"
            "优先使用架构、总参数量、激活参数量、精度与并行配置；若保留 model ID 仅用于分组切分与审计。",
            "",
            "## 当前最重要缺口",
            "",
            xpre["most_important_gap"],
            "",
            "建议下一阶段先生成版本化的 `planned_workload_features.parquet`：按 run 保存计划请求数、"
            "输入 token 总量及均值/P50/P90/P95/最大值、输出上限及上限总量，并附请求清单、数据集、"
            "tokenizer、模板和 benchmark commit 的哈希。实际输出 token 不可冒充事前特征。",
        ]
    )
    write(ROOT / "docs" / "x_pre_feature_catalog.md", "\n".join(lines))


def render_summary(full: dict[str, Any], comparison: dict[str, Any], workload: dict[str, Any]) -> None:
    stable = full["stability"]["stable"]
    total = full["row_count"]
    cross = full["cross_coverage"]
    lines = [
        "# ML.ENERGY Phase 1.5 数据审计与建模可行性判断",
        "",
        "## 结论先行",
        "",
        "**判定：B——有条件可继续，但必须先补齐事前工作量特征，再进入正式建模。**",
        "",
        "ML.ENERGY V3 的文本 LLM inference 汇总数据在内部一致性、GPU/模型/配置覆盖和目标口径上"
        "足以支持“受限 benchmark 域内”的下一阶段实验设计；但现有 parquet 没有保存可直接使用的"
        "事前输入 token 工作量/分布，输出上限也不能代表实际输出需求，因此尚不足以直接建立可信的"
        "“任务事前特征 → GPU 能耗/功率”正式模型。",
        "",
        "## 主要发现",
        "",
        f"- 源 parquet 共 {full['source_population']['parquet_all_llm_and_mllm_rows']} 行、37 列；"
        f"排除 {full['source_population']['excluded_mllm_rows']} 条 image/video MLLM 后，文本 LLM 为 {total} 行。",
        f"- 稳定 {stable} 行（{stable / total:.1%}），不稳定 {full['stability']['unstable']} 行。"
        f"不稳定原因：低 batch 利用率 {full['stability']['unstable_reason_categories'].get('low_batch_utilization', 0)}、"
        f"稳态过短 {full['stability']['unstable_reason_categories'].get('short_steady_state', 0)}、"
        f"由不稳定 batch 级联 {full['stability']['unstable_reason_categories'].get('cascade_from_unstable_batch', 0)}。",
        f"- 三任务稳定数：GPQA {full['stability']['by_task']['gpqa']['stable']}/247、"
        f"LM Arena {full['stability']['by_task']['lm-arena-chat']['stable']}/388、"
        f"Sourcegraph FIM {full['stability']['by_task']['sourcegraph-fim']['stable']}/59。",
        f"- 硬件仅 H100/B200；27 个模型形成 {cross['observed_task_gpu_model_cells']} 个实际 task×GPU×model 单元，"
        f"远少于朴素全笛卡尔 {cross['naive_full_cartesian_cells']}，属于非平衡设计，不能随机拆行后宣称跨模型泛化。",
        f"- 第一阶段 leaderboard 的 565 条稳定记录与完整 parquet 的稳定文本投影差异为 "
        f"{comparison['multiset_difference']}。",
        "- 37 列无 Arrow null、无完全重复行；8 字段表观键的 5 组重复由 `num_request_repeats` 区分。"
        "全体 seed 均为 48105，不能估计跨 seed 测量方差。",
        "- `E=P×T` 及另外三组派生恒等式在 694 行全部通过；这些是聚合 GPU 侧稳态指标，"
        "不是 CPU、整机或节点剩余功耗。",
        f"- 11 个代表 run 覆盖全部预设维度；只保留 {sum(r['metadata_bytes'] for r in load('selected_runs.json')['runs'])} 字节"
        "的前缀元数据，timeline 与 Prometheus 均未用于审计。",
        "",
        "## 限制",
        "",
        "- `task` 标签和 `max_output_tokens` 不能充分表征输入长度分布、生成难度或预期输出需求。",
        "- `total_input_tokens` 可以在执行前重构，但现有值来自结果文件；不独立重构就会形成时间边界污染。",
        "- 实际输出 token、吞吐、时长、ITL、完成数、平均 batch、稳定性标记及所有能耗/功率结果均是泄漏或目标。",
        "- 只有两个 GPU 代际、固定 benchmark 任务和单一 seed；外推到其他 GPU、自由用户流量或节点级功耗没有证据。",
        "- 前两个完整结果文件曾短暂暴露内嵌 timeline；发现后立即删除，最终仅保留并使用 timeline 之前的 16.6 KB 配置/汇总头部。",
        "",
        "## 下一阶段建议（此处停止，不训练模型）",
        "",
        "1. 冻结每个 run 的请求清单、数据集 revision、benchmark commit、模型 tokenizer revision 与 prompt/FIM 模板。",
        "2. 独立预计算事前输入 token 总量及分布统计、计划请求数、输出 token 上限总量；保留完整来源哈希。",
        "3. 新增跨 seed/重复测量以估计噪声；稳定样本作为主分析集，非稳定样本仅用于运行域边界与质量诊断。",
        "4. 下一阶段若试建模型，必须按 model/task/GPU 进行分组留出，先做物理基线与消融；"
        "不得用随机行切分掩盖同模型同配置邻近点泄漏。",
        "5. 若研究目标转为节点或设施功耗，需另采 CPU/内存/风扇/网络与节点基线；不能从本 GPU 侧数据推断。",
        "",
        "Phase 1.5 到此停止。",
    ]
    write(ROOT / "reports" / "phase1_5_audit_summary.md", "\n".join(lines))


def main() -> None:
    full = load("full_parquet_audit.json")
    comparison = load("leaderboard_comparison.json")
    workload = load("workload_field_audit.json")
    render_field_dictionary(full)
    render_workload_matrix(workload)
    render_x_pre(workload)
    render_summary(full, comparison, workload)
    print("Rendered field dictionary, workload matrix, X_pre catalog and summary")


if __name__ == "__main__":
    main()
