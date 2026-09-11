-- ML.ENERGY V3 LLM audit report snapshot.
-- Inputs were audited by scripts/audit_mlenergy_v3_llm.py and stored in
-- analysis/audit_results.json. These deterministic SQLite views reproduce the
-- exact rows embedded in reports/artifact.json; they do not query raw timelines.

CREATE TEMP VIEW headline AS
SELECT
  565 AS records,
  27 AS models,
  3 AS tasks,
  2 AS gpu_models,
  23 AS public_fields,
  0 AS null_cells,
  108 AS paired_keys,
  0.2378854626 AS paired_rate;

CREATE TEMP VIEW task_gpu AS
SELECT 'gpqa' AS task, 'H100' AS gpu, 81 AS count
UNION ALL SELECT 'gpqa', 'B200', 106
UNION ALL SELECT 'lm-arena-chat', 'H100', 177
UNION ALL SELECT 'lm-arena-chat', 'B200', 151
UNION ALL SELECT 'sourcegraph-fim', 'H100', 16
UNION ALL SELECT 'sourcegraph-fim', 'B200', 34;

CREATE TEMP VIEW num_gpus AS
SELECT '1' AS num_gpus, 250 AS count
UNION ALL SELECT '2', 104
UNION ALL SELECT '4', 83
UNION ALL SELECT '8', 128;

CREATE TEMP VIEW quality_checks AS
SELECT '公开字段空值' AS "check", 12995 AS checked, 0 AS failures, '23×565 个单元均非空' AS interpretation
UNION ALL SELECT '完全重复行', 565, 0, '没有逐字段完全相同的行'
UNION ALL SELECT '候选配置键重复组', 562, 3, '6 行；需用 parquet 的 seed/重复轮次解释'
UNION ALL SELECT '功率代数恒等式', 565, 0, '最大相对误差 2.48e-16'
UNION ALL SELECT '单请求能耗派生恒等式', 565, 0, '证实该列由实际输出长度派生'
UNION ALL SELECT 'ITL 分位顺序', 565, 0, 'median ≤ p90 ≤ p95 ≤ p99'
UNION ALL SELECT 'batch 利用率低于 0.85', 565, 0, '最低 0.8540；仅复核公开可见稳定条件';

CREATE TEMP VIEW numeric_ranges AS
SELECT 'total_params_billions' AS metric, 8.0 AS minimum, 32.0 AS median, 671.0 AS maximum, '十亿参数' AS unit
UNION ALL SELECT 'activated_params_billions', 3.0, 17.0, 405.0, '十亿参数'
UNION ALL SELECT 'avg_power_watts', 293.91, 1021.969, 7710.25, 'W（跨卡聚合）'
UNION ALL SELECT 'energy_per_token_joules', 0.02845, 0.46724, 34.3336, 'J/output token'
UNION ALL SELECT 'energy_per_request_joules', 18.472, 444.043, 117113.047, 'J/request（估算）'
UNION ALL SELECT 'avg_output_len', 229.622, 861.154, 11860.874, '实际 output tokens';

CREATE TEMP VIEW feature_decisions AS
SELECT '候选事前输入' AS class, 'task；architecture；total/activated params；weight_precision' AS fields, '可用，但 task 只是粗粒度工作负载代理' AS decision
UNION ALL SELECT '候选事前输入', 'gpu_model；num_gpus；max_num_seqs；TP/EP/DP', '可用；max_num_seqs 不是实际平均 batch'
UNION ALL SELECT '首选目标', 'energy_per_token_joules；avg_power_watts', '二者分开建模；功率为跨卡聚合'
UNION ALL SELECT '条件目标', 'energy_per_request_joules；steady_state_energy_joules', '需计划工作量/时长；前者当前为派生估算'
UNION ALL SELECT '明确泄漏', 'avg_output_len；avg_batch_size；throughput；ITL；duration；实际 token/requests', '全部禁作事前输入'
UNION ALL SELECT '标识风险', 'model_id；nickname', '仅分组、去重和外推验证，不作默认输入'
UNION ALL SELECT '诊断/追溯', 'seed；num_request_repeats；is_stable/reason；paths', '质量控制，不作主预测特征';

CREATE TEMP VIEW limitations AS
SELECT 'gated parquet 未授权' AS gap, '无法实审稳态总能量/时长、seed、重复轮次和稳定原因' AS impact, '授权后只下载 runs/llm.parquet' AS next_check
UNION ALL SELECT '缺少输入长度与计划输出/请求量', '无法形成真正的任务事前工作量描述，也无法稳健预测总能耗', '从工作负载配置提取计划特征'
UNION ALL SELECT '仅 3 任务、27 模型、H100/B200', '对新任务、新模型、新 GPU 的外推证据薄弱', '采用留一组外验证并报告支持域'
UNION ALL SELECT '公开快照只含稳定记录', '存在选择偏差，且无法分析失败/不稳定边界', '用 parquet 的稳定标志复核筛选前后分布'
UNION ALL SELECT '只有 GPU 侧稳态口径', '不能回答 CPU、节点/PDU 或 Node Residual', '另立阶段设计独立节点侧测量，不在当前范围'
UNION ALL SELECT '运行环境/硬件细节有限', '功率上限、频率、拓扑、服务器与框架版本变化可能造成域偏移', '补齐可审计的执行前环境元数据';
