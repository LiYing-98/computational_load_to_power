# Phase5 特征语义审计

主cohort保持565条stable GPU-side inference记录。详细static未伪造Llama信息。Hybrid的dense/MoE transformer layer composition为结构N/A：显式0加applicability指示，其原始统计位于只读Step6。MLA/hybrid不套GQA公式；GQA runtime KV dtype未知时L4不可用。Text proxy用冻结原始输入列表并排除completion；长度与UTF-8 bytes只表示输入文本，不代表模型token。主层无task标签。

覆盖和N/A/unknown计数见analysis CSV；未把较小support的绝对MAE当作特征增益。
