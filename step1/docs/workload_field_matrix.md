# 三类文本 LLM inference 工作量字段矩阵

官方 benchmark 源码固定在提交 `c1d6557fbf7c7c495749b0fcf649f71774eabde9`；代表性 metadata 共 11 份，保留 timeline payload 为 0。

| 任务 | 官方数据集 | split/subset | 唯一请求数 | 代表样本重复次数 | 单请求输出上限 | endpoint | 输入 token 事前状态 |
|---|---|---|---:|---|---:|---|---|
| `gpqa` | `Idavidrein/gpqa` | `train / gpqa_diamond` | 198 | [1, 6] | 32768 | `openai-chat` | 未预存；仅在冻结请求、tokenizer 与模板后可事前推导 |
| `lm-arena-chat` | `lmarena-ai/arena-human-preference-100k` | `train` | 1024 | [1, 2] | 4096 | `openai-chat` | 未预存；仅在冻结请求、tokenizer 与模板后可事前推导 |
| `sourcegraph-fim` | `sourcegraph/context-aware-fim-code-completions` | `train` | 1024 | [1, 3, 4] | 2048 | `openai` | 未预存；仅在冻结请求、tokenizer 与模板后可事前推导 |

统一流量配置：代表样本中 `request_rate=inf`、`burstiness=1.0`、`max_concurrency=null`、`max_num_batched_tokens=null`。源码默认采样参数为 `top_p=0.95`、`temperature=0.8`、`ignore_eos=false`；这些参数未写入所审计的结果头部，应在下一阶段配置清单中显式固化。

## 代表性结果头部字段

| JSON 路径 | 分类 | 可进入 `X_pre` | 出现数 | 任务覆盖 | 示例 |
|---|---|---:|---:|---|---|
| `burstiness` | `pre_known` | 是 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [1.0] |
| `completed` | `post_run_leakage` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [198, 1024] |
| `date` | `diagnostic` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | ["20251125-162658", "20251122-091553"] |
| `duration` | `post_run_leakage` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [933.1770403385162, 413.48613476753235] |
| `endpoint_type` | `pre_known` | 是 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | ["openai-chat", "openai"] |
| `entire_benchmark_measurement.cpu_energy` | `out_of_scope` | 否 | 8 | gpqa, lm-arena-chat, sourcegraph-fim | [null] |
| `entire_benchmark_measurement.cpu_energy.0` | `out_of_scope` | 否 | 3 | gpqa, lm-arena-chat, sourcegraph-fim | [115219.83619999999, 49299.756788] |
| `entire_benchmark_measurement.cpu_energy.1` | `out_of_scope` | 否 | 3 | gpqa, lm-arena-chat, sourcegraph-fim | [2261026.0, 1477446.0] |
| `entire_benchmark_measurement.cpu_energy.2` | `out_of_scope` | 否 | 3 | gpqa, lm-arena-chat, sourcegraph-fim | [111533.217037, 47061.110479999974] |
| `entire_benchmark_measurement.dram_energy` | `out_of_scope` | 否 | 8 | gpqa, lm-arena-chat, sourcegraph-fim | [null] |
| `entire_benchmark_measurement.dram_energy.0` | `out_of_scope` | 否 | 3 | gpqa, lm-arena-chat, sourcegraph-fim | [2226.8235669999995, 940.1957709999915] |
| `entire_benchmark_measurement.dram_energy.2` | `out_of_scope` | 否 | 3 | gpqa, lm-arena-chat, sourcegraph-fim | [2287.174370999999, 841.6839980000004] |
| `entire_benchmark_measurement.gpu_energy.0` | `diagnostic_outcome` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [441346.70299999416, 196137.06700000167] |
| `entire_benchmark_measurement.gpu_energy.1` | `diagnostic_outcome` | 否 | 8 | gpqa, lm-arena-chat, sourcegraph-fim | [186078.16899999976, 2167149.5629999936] |
| `entire_benchmark_measurement.gpu_energy.2` | `diagnostic_outcome` | 否 | 6 | gpqa, lm-arena-chat, sourcegraph-fim | [95289.61400008202, 198908.27500000596] |
| `entire_benchmark_measurement.gpu_energy.3` | `diagnostic_outcome` | 否 | 6 | gpqa, lm-arena-chat, sourcegraph-fim | [109335.94199991226, 209581.31899999082] |
| `entire_benchmark_measurement.gpu_energy.4` | `diagnostic_outcome` | 否 | 3 | lm-arena-chat, sourcegraph-fim | [200314.53499999642, 209079.88599999994] |
| `entire_benchmark_measurement.gpu_energy.5` | `diagnostic_outcome` | 否 | 3 | lm-arena-chat, sourcegraph-fim | [209379.68500000238, 215790.1519999951] |
| `entire_benchmark_measurement.gpu_energy.6` | `diagnostic_outcome` | 否 | 3 | lm-arena-chat, sourcegraph-fim | [209083.8490000069, 215215.16899999976] |
| `entire_benchmark_measurement.gpu_energy.7` | `diagnostic_outcome` | 否 | 3 | lm-arena-chat, sourcegraph-fim | [199651.26000000536, 210295.83399999887] |
| `entire_benchmark_measurement.soc_energy` | `out_of_scope` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [null] |
| `entire_benchmark_measurement.time` | `post_run_leakage` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [933.1733644008636, 413.47928953170776] |
| `gpu_model` | `pre_known` | 是 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | ["B200", "H100"] |
| `max_concurrency` | `pre_known` | 是 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [null] |
| `max_num_batched_tokens` | `pre_known` | 是 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [null] |
| `max_num_seqs` | `pre_known` | 是 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [1024, 128] |
| `max_output_tokens` | `pre_known` | 是 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [32768, 2048] |
| `model_id` | `identifier_risk` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | ["Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", "Qwen/Qwen3-14B"] |
| `num_gpus` | `pre_known` | 是 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [1, 4] |
| `num_prompts` | `pre_derivable` | 是 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [198, 1024] |
| `num_request_repeats` | `pre_known` | 是 | 10 | gpqa, lm-arena-chat, sourcegraph-fim | [1, 2] |
| `num_unique_prompts` | `pre_known` | 是 | 10 | gpqa, lm-arena-chat, sourcegraph-fim | [1024, 198] |
| `output_throughput` | `post_run_leakage` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [1554.6331910113556, 5100.4950896012515] |
| `request_rate` | `pre_known` | 是 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | ["inf"] |
| `request_throughput` | `post_run_leakage` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [0.2121783878525067, 2.8731314066139366] |
| `seed` | `diagnostic` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [48105] |
| `steady_state_duration` | `post_run_leakage` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [298.17371940612793, 60.085633754730225] |
| `steady_state_energy` | `target` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [152443.9129999876, 68434.66600000858] |
| `steady_state_energy_per_token` | `target` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [0.18257543199465798, 0.09540020046199459] |
| `steady_state_measurement.cpu_energy` | `out_of_scope` | 否 | 8 | gpqa, lm-arena-chat, sourcegraph-fim | [null] |
| `steady_state_measurement.cpu_energy.0` | `out_of_scope` | 否 | 3 | gpqa, lm-arena-chat, sourcegraph-fim | [16987.293670000014, 14488.053242000009] |
| `steady_state_measurement.cpu_energy.1` | `out_of_scope` | 否 | 3 | gpqa, lm-arena-chat, sourcegraph-fim | [298687.0, 444321.0] |
| `steady_state_measurement.cpu_energy.2` | `out_of_scope` | 否 | 3 | gpqa, lm-arena-chat, sourcegraph-fim | [15741.61656200001, 13729.461800999998] |
| `steady_state_measurement.dram_energy` | `out_of_scope` | 否 | 8 | gpqa, lm-arena-chat, sourcegraph-fim | [null] |
| `steady_state_measurement.dram_energy.0` | `out_of_scope` | 否 | 3 | gpqa, lm-arena-chat, sourcegraph-fim | [320.38640699999814, 311.9483640000035] |
| `steady_state_measurement.dram_energy.2` | `out_of_scope` | 否 | 3 | gpqa, lm-arena-chat, sourcegraph-fim | [300.6674600000006, 245.03343200000018] |
| `steady_state_measurement.gpu_energy.0` | `target` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [152443.9129999876, 34833.91400000453] |
| `steady_state_measurement.gpu_energy.1` | `target` | 否 | 8 | gpqa, lm-arena-chat, sourcegraph-fim | [33600.75200000405, 2106646.5130000114] |
| `steady_state_measurement.gpu_energy.2` | `target` | 否 | 6 | gpqa, lm-arena-chat, sourcegraph-fim | [30749.578000068665, 84897.29999999702] |
| `steady_state_measurement.gpu_energy.3` | `target` | 否 | 6 | gpqa, lm-arena-chat, sourcegraph-fim | [35428.0720000267, 88971.875] |
| `steady_state_measurement.gpu_energy.4` | `target` | 否 | 3 | lm-arena-chat, sourcegraph-fim | [85345.54999999702, 61028.72500000149] |
| `steady_state_measurement.gpu_energy.5` | `target` | 否 | 3 | lm-arena-chat, sourcegraph-fim | [88686.13600000739, 61845.89999999851] |
| `steady_state_measurement.gpu_energy.6` | `target` | 否 | 3 | lm-arena-chat, sourcegraph-fim | [88566.98900000751, 61604.13800000399] |
| `steady_state_measurement.gpu_energy.7` | `target` | 否 | 3 | lm-arena-chat, sourcegraph-fim | [84835.96700000763, 61265.84499999881] |
| `steady_state_measurement.soc_energy` | `out_of_scope` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [null] |
| `steady_state_measurement.time` | `post_run_leakage` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [298.17371940612793, 60.085633754730225] |
| `total_input_tokens` | `post_run_leakage` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [44473, 1306650] |
| `total_output_tokens` | `post_run_leakage` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [1450748, 2108984] |
| `total_token_throughput` | `post_run_leakage` | 否 | 11 | gpqa, lm-arena-chat, sourcegraph-fim | [1602.2908144606713, 5728.448431122555] |

## 关键解释

- `num_prompts` 可在执行前由 `num_unique_prompts × num_request_repeats` 得到。
- 结果文件里的 `total_input_tokens` 虽可理论上提前重构，但该列本身是执行后记录；建模时必须使用独立的事前重构值，不能直接回填结果列。
- `max_output_tokens` 只是上限，不是实际或期望输出长度；`total_output_tokens`、吞吐、时长、完成数、ITL、平均 batch 均是执行后泄漏变量。
- Sourcegraph FIM 的 prompt 格式随模型 tokenizer 类型变化；LM Arena 多轮拼接、GPQA 选项随机化也受固定 seed 与实现版本约束，所以必须冻结源码、数据集 revision、请求清单和 tokenizer revision。
