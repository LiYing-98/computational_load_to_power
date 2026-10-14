# Phase5 compact physics formulas

主L4仅支持冻结静态信息和benchmark workload已知的GQA dense/MoE；MLA/hybrid不支持，GPT-OSS KV运行dtype未知也不伪造。所有输入事前可得，无actual output token。

记N=layers、H=hidden、A=query heads、K=KV heads、D=head dimension、I=dense或MoE expert width、E=active experts（dense=1）、g=gated的3个线性项或non-gated的2项、b=冻结runtime dtype bits/8；μ=benchmark mean length、m2=E[L²]。

- projection/request = 2 N H D (2A + 2K) μ。
- attention/request = 4 N A D m2。沿用既有full-square FLOP proxy，非宣称实际causal kernel精确操作数；忽略sliding window、softmax、normalization等。
- MLP/request = 2 g N H I E μ。主支持集MoE为全部层同类型且无已知shared expert的Qwen；若混合dense/MoE或shared项未知则排除，不能套统一层公式。
- prefill total = projection + attention + MLP；是确定性加和，保留机器可读派生值但从正式X删除。
- KV bytes/request = 2 N K D b L；分别用mean、p95和max输入长度。
- one-token decode representative context = projection(μ=1) + MLP(μ=1) + 4 N A D (μ+1)。无实际输出量或时长。
- weight residency来源于L1，L4不重复添加完全相同字段。

这些量为理论intensity proxy，非真实FLOP/s、KV占用或通信时间。平均单GPU目标表示聚合功率/卡数，非逐卡测量。旧B5比率仅敏感性分支。
