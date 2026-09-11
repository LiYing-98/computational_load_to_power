# `X_pre` 特征目录与泄漏边界

原则：预测时只能使用任务发起前已经给定，或能从冻结配置与输入确定性计算出的信息。任何来自实际生成、调度、吞吐、时长或能源测量的量均不得进入输入。

## 可直接使用的事前字段

- `task`
- `architecture`
- `total_params_billions`
- `activated_params_billions`
- `weight_precision`
- `gpu_model`
- `num_gpus`
- `max_num_seqs`
- `tensor_parallel`
- `expert_parallel`
- `data_parallel`
- `num_unique_prompts`
- `num_request_repeats`
- `max_num_batched_tokens`
- `request_rate`
- `burstiness`
- `max_concurrency`
- `max_output_tokens`
- `endpoint_type`

## 可事前推导的字段

- `planned_request_count = num_unique_prompts * num_request_repeats`
- `planned_output_token_upper_bound = planned_request_count * max_output_tokens`
- `planned_total_input_tokens (only after frozen-request/tokenizer reconstruction)`
- `planned_input_length distribution summaries (same precondition)`
- `GPU-normalized model size and parallelism ratios from physical metadata`

## 禁止作为输入的执行后字段

- `completed`
- `duration`
- `entire_benchmark_measurement.gpu_energy.0`
- `entire_benchmark_measurement.gpu_energy.1`
- `entire_benchmark_measurement.gpu_energy.2`
- `entire_benchmark_measurement.gpu_energy.3`
- `entire_benchmark_measurement.gpu_energy.4`
- `entire_benchmark_measurement.gpu_energy.5`
- `entire_benchmark_measurement.gpu_energy.6`
- `entire_benchmark_measurement.gpu_energy.7`
- `entire_benchmark_measurement.time`
- `output_throughput`
- `request_throughput`
- `steady_state_duration`
- `steady_state_energy`
- `steady_state_energy_per_token`
- `steady_state_measurement.gpu_energy.0`
- `steady_state_measurement.gpu_energy.1`
- `steady_state_measurement.gpu_energy.2`
- `steady_state_measurement.gpu_energy.3`
- `steady_state_measurement.gpu_energy.4`
- `steady_state_measurement.gpu_energy.5`
- `steady_state_measurement.gpu_energy.6`
- `steady_state_measurement.gpu_energy.7`
- `steady_state_measurement.time`
- `total_input_tokens`
- `total_output_tokens`
- `total_token_throughput`

## 标识符风险

`model_id`/`nickname` 虽在执行前已知，但直接编码会让模型记忆具体模型身份。优先使用架构、总参数量、激活参数量、精度与并行配置；若保留 model ID 仅用于分组切分与审计。

## 当前最重要缺口

No persisted pre-execution workload token-volume/distribution artifact. Observed total_input_tokens and all generated-output quantities are post-run values.

建议下一阶段先生成版本化的 `planned_workload_features.parquet`：按 run 保存计划请求数、输入 token 总量及均值/P50/P90/P95/最大值、输出上限及上限总量，并附请求清单、数据集、tokenizer、模板和 benchmark commit 的哈希。实际输出 token 不可冒充事前特征。
