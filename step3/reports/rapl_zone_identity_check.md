# RAPL zone identity time-box

结论：zone 1 状态为 **unresolved**。没有找到把整数索引 1 直接映射到 RAPL `name` 的字符串证据，因此继续使用中性标签 **platform-like diagnostic**；不能根据功率量级把它命名为 `psys`、package 或整机功率。

本检查只读取既有的 694 份轻量结果头、Phase 1.6 审计产物、固定 Zeus 0.13.1 源码和固定 benchmark 源码；没有新增下载，没有读取完整 timeline，也没有下载 Prometheus。

## 已检查证据

| Source/manifest | SHA-256 | Files inspected | Direct mappings |
|---|---|---:|---:|
| `step2/data/zeus-0.13.1/zeus/device/cpu/rapl.py` | `969cba2ecc862f37477e878d117a8d385af6c9e38aab6394a0fb7a203ebe821d` | 1 | 0 |
| `step2/data/zeus-0.13.1/zeus/monitor/energy.py` | `e59ba55570c6f55afff63fdb7fac4ff77fff02acb79ef0c4cb724503a9570248` | 1 | 0 |
| `step2/data/official_source/benchmark.py` | `c8c0f55d5d542c2da851aa0f9023c2f263f383587d598b16affce54b3e8035c1` | 1 | 0 |
| `step2/analysis/cpu_dram_audit.json` | `e10de1aa715f27e0a7bd4ec127a691f1d6badd63e522faf2b5639280ef46d3e4` | 1 | 0 |
| `step2/docs/cpu_dram_measurement_boundary.md` | `e959258ded7a46090729dc957f50a5bd455dad04066ce2c821c7bc12550e0e7d` | 1 | 0 |
| `step2/reports/phase1_6_audit_summary.md` | `7890ef2488246d8914b31c61d21b4abc46378563105bc1bda998437e1587554b` | 1 | 0 |
| `step2/analysis/result_metadata_manifest.json` | `2d71cb2288084bd6e45b90967d8d489e76683653309a5b669d3b3ee36b7ae9ed` | 694 | 0 |

## 数值证据的限定用途

Phase 1.6 中 zone 1 对应的无 DRAM 顶层区功率统计为 median=5932.083654177734 W、min=2665.8009357511733 W、max=8615.443867600654 W。该量级只支持‘需要进一步核实测量边界’的诊断，不构成名称证据。

下一阶段若要解析身份，应在采集时同时保存 `/sys/class/powercap/intel-rapl/intel-rapl:*/name`、路径、socket/NUMA 拓扑和采集时间；在此之前 CPU 三域和 zone 1 不进入本阶段 GPU 目标或 X。
