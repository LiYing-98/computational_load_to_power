#!/usr/bin/env python3
"""Create the Phase 1.5 reproducible audit notebook from reviewed JSON outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
    cells = []
    cells.append(
        nbf.v4.new_markdown_cell(
            """# ML.ENERGY V3 Phase 1.5 — 完整汇总与工作量字段审计

## tl;dr

**判定 B：有条件可继续。** 694 条文本 LLM inference 汇总记录内部一致、稳定子集与 Phase 1 完全一致；但正式建模前必须独立重构并冻结事前输入 token 工作量/分布。本文不训练模型，只复现审计证据。"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## Context & Methods

- 范围仅含 GPQA、LM Arena Chat、Sourcegraph FIM 三类文本 LLM inference。
- 指标是稳态窗口内所有 GPU 的聚合 GPU 侧口径。
- 笔记本只读取 Git 中的审计 JSON，不读取 gated parquet、Prometheus 或 timeline。
- 输入变量按执行前已知/可推导与执行后泄漏严格分开。"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            r"""import json
from pathlib import Path
from IPython.display import Markdown, display

ROOT = Path.cwd()
if not (ROOT / 'step1' / 'analysis').exists():
    ROOT = ROOT.parent.parent

def load(name):
    return json.loads((ROOT / 'step1' / 'analysis' / name).read_text(encoding='utf-8'))

full = load('full_parquet_audit.json')
comparison = load('leaderboard_comparison.json')
selected = load('selected_runs.json')
workload = load('workload_field_audit.json')

assert full['row_count'] == 694
assert full['stability']['stable'] == 565
assert comparison['multiset_difference'] == 0
assert workload['decision']['code'] == 'B'
assert workload['scope']['timeline_payloads_retained'] == 0
print('Reviewed outputs loaded and headline assertions passed.')"""
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## Data"))
    cells.append(
        nbf.v4.new_code_cell(
            r"""summary = f'''| 指标 | 值 |
|---|---:|
| parquet 全部 LLM+MLLM 行 | {full['source_population']['parquet_all_llm_and_mllm_rows']} |
| 排除 MLLM 行 | {full['source_population']['excluded_mllm_rows']} |
| 文本 LLM 行 | {full['row_count']} |
| 稳定 / 非稳定 | {full['stability']['stable']} / {full['stability']['unstable']} |
| 模型数 | {full['cross_coverage']['models']} |
| task×GPU×model 单元 | {full['cross_coverage']['observed_task_gpu_model_cells']} |
| 代表性 metadata | {selected['selected_count']} |
| 保留 timeline | {workload['scope']['timeline_payloads_retained']} |
'''
display(Markdown(summary))"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            r"""import matplotlib.pyplot as plt

tasks = ['gpqa', 'lm-arena-chat', 'sourcegraph-fim']
stable = [full['stability']['by_task'][t]['stable'] for t in tasks]
unstable = [full['stability']['by_task'][t]['unstable'] for t in tasks]

fig, ax = plt.subplots(figsize=(8.2, 4.2))
ax.bar(tasks, stable, color='#2176AE', label='Stable')
ax.bar(tasks, unstable, bottom=stable, color='#F4A261', label='Unstable')
ax.set_ylabel('Run rows')
ax.set_title('Text-LLM stability by benchmark task')
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
plt.show()"""
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## Results"))
    cells.append(
        nbf.v4.new_code_cell(
            r"""reason_lines = '\n'.join(
    f"- `{reason}`: {count}" for reason, count in full['stability']['unstable_reason_categories'].items()
)
display(Markdown('### 非稳定原因\n\n' + reason_lines))

checks = full['consistency']
check_table = '| 校验 | 检查行 | 失败行 | 最大相对误差 |\n|---|---:|---:|---:|\n'
for name, result in checks.items():
    check_table += f"| `{name}` | {result['checked']} | {result['failed']} | {result.get('maximum_relative_error', '—')} |\n"
display(Markdown('### 一致性校验\n\n' + check_table))"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            r"""header = '| task | GPU | architecture | scale | GPUs | max_num_seqs | TP/EP/DP |\n|---|---|---|---|---:|---:|---|\n'
body = ''
for run in selected['runs']:
    body += (
        f"| {run['task']} | {run['gpu_model']} | {run['architecture']} | {run['model_scale']} | "
        f"{run['num_gpus']} | {run['max_num_seqs']} | "
        f"{run['tensor_parallel']}/{run['expert_parallel']}/{run['data_parallel']} |\n"
    )
display(Markdown('### 代表性 run 覆盖\n\n' + header + body))"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            r"""from collections import Counter

categories = Counter(item['category'] for item in workload['field_audit'].values())
display(Markdown('### 结果头部字段分类\n\n' + '\n'.join(f'- `{k}`: {v}' for k, v in sorted(categories.items()))))

matrix = '| task | unique prompts | repeats in sample | max output cap | endpoint | source check |\n|---|---:|---|---:|---|---|\n'
for task, spec in workload['task_matrix'].items():
    matrix += (
        f"| {task} | {spec['num_unique_prompts']} | {spec['observed_repeat_counts']} | "
        f"{spec['max_output_tokens']} | {spec['endpoint_type']} | {workload['cross_checks'][task]['status']} |\n"
    )
display(Markdown('### 工作量配置交叉核对\n\n' + matrix))"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. `E=P×T` 等四组恒等式在 694 行上全部通过，稳定子集与 Phase 1 投影差异为 0。
2. `num_request_repeats` 是计划工作量变量，不是统计学上的独立重复；全体 seed 固定为 48105。
3. 允许进入 `X_pre` 的是任务、模型物理元数据、GPU/卡数/并行与运行前流量配置。
4. 实际 input/output tokens、吞吐、时长、ITL、平均 batch、稳定性与能源测量禁止作为输入。
5. 最大缺口是未持久化的事前输入 token 总量/分布；先补齐该表，再讨论正式模型。

**最终判定：B（有条件可继续）；Phase 1.5 到此停止。**"""
        )
    )

    nb["cells"] = cells
    output = Path("step1/notebooks/phase1_5_complete_audit.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
