# AI算力任务到功率曲线建模：阶段性课题认知

## 1. 研究背景与目标

本课题面向 AI 算力任务与电力负荷之间的跨域映射问题，目标是建立从**任务侧特征、执行配置与硬件侧特征**出发，对 AI 计算任务的**总能耗、任务持续时间及节点级时序功率曲线**进行预测与生成的统一建模框架。

研究最终希望回答以下核心问题：

> 给定一个 AI 算力任务及其运行配置和硬件平台，在任务实际执行前，能否预测其将产生的总能耗、持续时间以及完整功率曲线？

对应的总体映射关系可表示为：

\[
X=
\left[
X_{\mathrm{task}},
X_{\mathrm{architecture}},
X_{\mathrm{execution}},
X_{\mathrm{hardware}}
\right]
\rightarrow
P(t)
\]

其中：

- \(X_{\mathrm{task}}\)：任务类型、任务规模及输入输出工作量等特征；
- \(X_{\mathrm{architecture}}\)：模型结构特征；
- \(X_{\mathrm{execution}}\)：并行策略、精度、框架及运行配置；
- \(X_{\mathrm{hardware}}\)：GPU、节点、互联等硬件平台特征；
- \(P(t)\)：任务执行过程中形成的节点级功率时间序列。

本课题的核心不是简单构建一个“GPU 功率预测模型”，而是建立：

> **AI 算力任务 → 计算执行特征 → 能耗与功率负荷**

之间具有物理解释性、跨任务与跨硬件泛化能力的映射关系。

---

# 2. 关于“任务 → 功率曲线”直接映射的基本认识

理论上，只要任务侧、执行侧和硬件侧输入信息足够完整，客观上一定存在：

\[
X \rightarrow P(t)
\]

的映射。

例如输入变量可以包括：

- 训练 / 推理；
- LLM / CV / Diffusion / 多模态等任务模态；
- Dense / MoE；
- 模型参数规模；
- hidden size；
- intermediate size；
- embedding size；
- attention head 数；
- KV head 数；
- layer 数；
- sequence length；
- batch size；
- precision；
- DP / TP / PP / EP；
- GPU 数量；
- GPU 型号；
- 显存容量；
- NVLink / PCIe / InfiniBand 等互联能力；
- CUDA、计算框架与算子优化方式；
- Power cap、DVFS 等运行约束。

这些变量在很大程度上已经决定：

1. 任务需要完成多少计算；
2. 需要进行多少显存访问；
3. 需要进行多少跨 GPU / 跨节点通信；
4. 不同执行阶段的持续时间；
5. 哪些阶段可能出现 compute-bound、memory-bound 或 communication-bound；
6. 最终形成何种功率水平与功率波动。

因此，从科学上看，**不存在必须先显式预测“执行阶段”再预测功率的必要性**。

---

# 3. 不采用强串联“阶段识别 → 资源状态 → 功率”模型的原因

一种直观的建模路径是：

\[
X
\rightarrow
\text{Execution Stage}
\rightarrow
\text{Resource State}
\rightarrow
P(t)
\]

例如分别预测：

\[
X
\xrightarrow{\text{Model 1}}
Stage
\]

\[
Stage
\xrightarrow{\text{Model 2}}
Resource
\]

\[
Resource
\xrightarrow{\text{Model 3}}
Power
\]

该方法具有较强物理解释性，但存在一个明显风险：

> **多个预测模型串联后，前一级预测误差会向后传播并被进一步放大。**

假设各级预测误差分别为：

\[
\epsilon_1,\epsilon_2,\epsilon_3
\]

则最终功率预测误差往往不仅取决于最后一级模型，而会受到所有中间预测误差的共同影响。

因此，目前阶段不建议将“阶段识别”和“资源状态预测”作为必须经过的硬串联预测层。

更合理的处理方式是：

> 将执行阶段、资源瓶颈、FLOPs、显存访问量、通信量等信息作为**物理先验、派生特征、辅助任务或形态约束**，而不是作为主预测链条中的必经中间预测变量。

---

# 4. 建议采用 Energy–Duration–Shape（EDS）分解框架

相比直接预测一个高维、变长度的功率时间序列，更适合当前课题的思路是将真实功率曲线分解为三个组成部分：

1. **Energy：总能耗 \(E\)**
2. **Duration：任务持续时间 \(T\)**
3. **Shape：归一化功率曲线 \(q(s)\)**

即：

\[
\boxed{
X
\rightarrow
\left(E,T,q(s)\right)
\rightarrow
P(t)
}
\]

该框架可以暂称为：

> **Task-conditioned Energy–Duration–Shape Power Modeling**

即“任务条件驱动的能耗—时长—形态功率建模框架”。

---

# 5. 为什么必须同时建模总能耗和持续时间

仅使用：

\[
X\rightarrow E
\]

再结合一条归一化功率曲线，并不能唯一恢复真实功率曲线。

原因在于总能耗满足：

\[
E=\int_0^T P(t)\,dt
\]

同一个总能耗可以对应完全不同的平均功率。

例如：

任务 A：

\[
E=10\ \mathrm{kWh},\quad T=1\ \mathrm{h}
\]

则：

\[
P_{\mathrm{avg}}=10\ \mathrm{kW}
\]

任务 B：

\[
E=10\ \mathrm{kWh},\quad T=10\ \mathrm{h}
\]

则：

\[
P_{\mathrm{avg}}=1\ \mathrm{kW}
\]

对于电力系统而言，这两类负荷具有完全不同的运行影响。

因此，任务功率曲线至少需要三个要素共同确定：

\[
\boxed{
E,\quad T,\quad q(s)
}
\]

---

# 6. EDS 分解的数学表达

定义归一化时间：

\[
s=\frac{t}{T},\qquad s\in[0,1]
\]

定义平均功率：

\[
P_{\mathrm{avg}}=\frac{E}{T}
\]

再定义归一化功率曲线：

\[
q(s)=\frac{P(t)}{P_{\mathrm{avg}}}
\]

即：

\[
q(s)=\frac{TP(t)}{E}
\]

要求：

\[
\int_0^1 q(s)\,ds=1
\]

则真实功率曲线可以由：

\[
\boxed{
P(t)=\frac{E}{T}q\left(\frac{t}{T}\right)
}
\]

恢复。

该表达式天然满足能量守恒：

\[
\int_0^T P(t)\,dt
=
E
\]

因此，整个任务功率曲线可理解为：

\[
\boxed{
\text{Power Curve}
=
\text{Amplitude Scale}
\times
\text{Shape}
}
\]

其中：

\[
\text{Amplitude Scale}=\frac{E}{T}
\]

而：

\[
q\left(\frac{t}{T}\right)
\]

决定任务执行过程中的相对功率形态。

---

# 7. EDS 三个预测对象的物理含义

## 7.1 总能耗模型

建立：

\[
E=f_E(X)
\]

回答：

> 一个 AI 算力任务完成全过程共需要消耗多少电能？

总能耗主要由以下因素决定：

- 总计算量；
- 总显存访问量；
- 总通信量；
- GPU 能效；
- CPU / 内存 / 网络辅助能耗；
- 并行效率；
- 计算精度；
- 任务规模；
- 软硬件优化水平。

总能耗模型对应任务的**能量尺度**。

---

## 7.2 持续时间模型

建立：

\[
T=f_T(X)
\]

回答：

> 一个 AI 算力任务在指定运行配置与硬件平台下需要运行多长时间？

持续时间本质上属于性能建模问题，受以下因素共同影响：

- 计算工作量；
- GPU 峰值算力；
- 实际计算利用率；
- memory bandwidth；
- communication bandwidth；
- tensor parallel efficiency；
- data parallel communication；
- pipeline bubble；
- MoE expert communication；
- batch size；
- sequence length；
- kernel efficiency；
- scheduler behavior。

总能耗与持续时间共同决定：

\[
P_{\mathrm{avg}}=\frac{E}{T}
\]

即任务的平均功率水平。

---

## 7.3 归一化功率形态模型

建立：

\[
q(s)=f_q(X)
\]

回答：

> 在去除任务总能耗和持续时间差异之后，任务执行过程中功率如何围绕平均功率发生变化？

该模型主要用于捕捉：

- 功率周期性；
- 峰谷结构；
- ramp；
- compute / communication 切换；
- 同步等待；
- I/O；
- Prefill / Decode；
- Forward / Backward；
- Optimizer；
- Diffusion iterative denoising 等时序特征。

归一化功率模型对应任务的**形态尺度**。

---

# 8. E、T 与 Shape 并不是物理独立变量

EDS 分解的本质是：

> **对输出空间进行解耦，而不是认为总能耗、持续时间和功率形态相互独立。**

三个模型仍然共享相同或部分重合的输入特征：

\[
E=f_E(X)
\]

\[
T=f_T(X)
\]

\[
q=f_q(X)
\]

例如改变 Tensor Parallel 数量，可能同时导致：

1. 总运行时间发生变化；
2. 总通信能耗发生变化；
3. 平均功率变化；
4. 功率曲线出现更加明显的计算—通信交替特征。

因此不能简单使用一个与任务配置无关的“标准功率模板”。

---

# 9. EDS 分解的重要优势：解决变长度时间序列问题

不同 AI 任务的持续时间可能存在巨大差异：

- 数秒；
- 数分钟；
- 数小时；
- 数天。

如果直接建立：

\[
X\rightarrow P(t)
\]

则输出时间序列长度不统一，会增加模型训练难度。

EDS 框架通过：

\[
s=\frac{t}{T}
\]

将所有任务映射至：

\[
s\in[0,1]
\]

例如统一采样为：

\[
q=
[q_1,q_2,\cdots,q_N]
\]

其中可以令：

\[
N=128,\ 256,\ 512
\]

从而将所有不同持续时间的任务转化为统一长度的归一化功率曲线。

预测完成后，通过：

\[
t_i=s_iT
\]

恢复真实时间轴。

因此，该框架天然适合处理 AI 任务持续时间跨度较大的问题。

---

# 10. Shape 模型不应一开始直接采用高维序列预测

初期不建议直接建立：

\[
X\rightarrow[q_1,q_2,\dots,q_{256}]
\]

的黑箱深度学习模型。

应首先分析归一化功率曲线是否存在低维结构。

可以尝试：

- Principal Component Analysis；
- Functional PCA；
- Fourier Basis；
- Wavelet Basis；
- Clustering；
- Autoencoder；
- Variational Autoencoder；
- 其他低维流形方法。

例如：

\[
q(s)
=
\bar q(s)
+
\sum_{k=1}^{K}a_k\phi_k(s)
\]

其中：

- \(\bar q(s)\)：平均功率形态；
- \(\phi_k(s)\)：第 \(k\) 个形态基函数；
- \(a_k\)：对应形态系数；
- \(K\ll N\)。

若只需要少量主成分即可解释绝大多数功率形态差异，则原始问题：

\[
X\rightarrow256\text{维功率序列}
\]

可以转化为：

\[
X\rightarrow
[a_1,a_2,\cdots,a_K]
\]

显著降低学习难度，同时提高模型可解释性和泛化能力。

---

# 11. 建议的任务侧输入变量体系

为避免把所有变量作为无结构特征直接输入模型，建议将输入变量划分为以下四类。

## 11.1 Workload-level：任务工作负载特征

包括但不限于：

- Training / Inference；
- Online inference / Offline inference；
- LLM / CV / Diffusion / Multimodal；
- 输入样本数量；
- 输入 token 数；
- 输出 token 数；
- sequence length；
- batch size；
- training steps；
- inference requests；
- diffusion steps；
- concurrency；
- request arrival rate 等。

---

## 11.2 Architecture-level：模型架构特征

包括：

- 总参数量；
- layer 数；
- hidden size；
- intermediate size；
- embedding size；
- attention head 数；
- KV head 数；
- MHA / GQA / MQA；
- Dense / MoE；
- expert 数量；
- active expert 数量；
- vocabulary size；
- model modality；
- vision encoder / language decoder 等多模态结构特征。

---

## 11.3 Execution-level：执行与并行配置特征

包括：

- Data Parallelism, DP；
- Tensor Parallelism, TP；
- Pipeline Parallelism, PP；
- Expert Parallelism, EP；
- 混合并行策略；
- local batch size；
- micro batch size；
- gradient accumulation；
- activation checkpointing；
- precision；
- quantization；
- CUDA Graph；
- FlashAttention；
- serving engine；
- training framework；
- inference framework；
- scheduler；
- kernel optimization。

该类变量是任务与硬件之间非常关键的桥梁。

---

## 11.4 Hardware-level：硬件平台特征

包括：

- GPU 型号；
- GPU 数量；
- GPU peak FLOPS；
- GPU memory capacity；
- memory bandwidth；
- TDP；
- SM 数量；
- Tensor Core 能力；
- GPU interconnect；
- NVLink bandwidth；
- PCIe generation；
- InfiniBand / Ethernet；
- 节点内 GPU 拓扑；
- CPU 型号；
- CPU 数量；
- system memory；
- storage；
- Power cap；
- DVFS 配置。

---

# 12. 不应只使用原始输入变量，还应构造物理派生特征

为了提高跨模型、跨配置和跨硬件的泛化能力，应进一步构造：

\[
X\rightarrow\Phi(X)
\]

其中：

\[
\Phi(X)
\]

表示由任务结构和硬件参数推导出的具有物理意义的特征。

建议重点构建：

## 12.1 计算量特征

例如：

\[
F_{\mathrm{forward}}
\]

\[
F_{\mathrm{backward}}
\]

\[
F_{\mathrm{optimizer}}
\]

\[
F_{\mathrm{prefill}}
\]

\[
F_{\mathrm{decode/token}}
\]

及总 FLOPs：

\[
F_{\mathrm{total}}
\]

---

## 12.2 显存与访存特征

例如：

\[
Bytes_{\mathrm{weights}}
\]

\[
Bytes_{\mathrm{activations}}
\]

\[
Bytes_{\mathrm{KV}}
\]

\[
Bytes_{\mathrm{optimizer}}
\]

及理论 memory traffic。

---

## 12.3 通信量特征

例如：

\[
V_{\mathrm{TP}}
\]

\[
V_{\mathrm{DP}}
\]

\[
V_{\mathrm{PP}}
\]

\[
V_{\mathrm{EP}}
\]

用于描述：

- All-Reduce；
- All-Gather；
- Reduce-Scatter；
- Point-to-point pipeline communication；
- MoE All-to-All 等通信负荷。

---

## 12.4 计算—访存关系特征

例如 Arithmetic Intensity：

\[
AI=\frac{\mathrm{FLOPs}}{\mathrm{Bytes}}
\]

可用于辅助判断：

- Compute-bound；
- Memory-bound；
- Communication-bound。

---

# 13. 物理派生特征与中间预测层的区别

物理派生特征：

\[
\Phi(X)
\]

与“先预测执行阶段再预测功率”存在本质区别。

如果 FLOPs、理论显存流量、通信量等特征可以由模型结构和执行配置通过公式或算法直接计算，则：

\[
\Phi(X)
\]

不是额外预测结果，不引入新的预测误差。

因此推荐实际模型输入为：

\[
\boxed{
[X,\Phi(X)]
}
\]

而不是只采用原始变量：

\[
X
\]

该方法兼顾：

- 数据驱动能力；
- 物理可解释性；
- 跨任务泛化；
- 跨硬件泛化；
- 降低机器学习模型学习难度。

---

# 14. 执行阶段识别在本课题中的正确定位

执行阶段仍然非常重要，但当前阶段不建议将其作为主预测链条中的硬中间层。

建议将其定位为：

## 14.1 机理解释工具

用于解释为什么某些功率曲线具有特定形态。

例如训练：

\[
Forward
\rightarrow
Backward
\rightarrow
Communication
\rightarrow
Optimizer
\]

LLM 推理：

\[
Prefill
\rightarrow
Decode
\]

Diffusion：

\[
Prompt/Condition Encoding
\rightarrow
Iterative Denoising
\rightarrow
Decode
\]

---

## 14.2 Shape 模型的结构先验

不同阶段决定不同的典型功率状态，可以帮助约束或解释：

\[
q(s)
\]

的局部形态。

---

## 14.3 辅助任务

未来可以考虑多任务学习：

\[
X\rightarrow
\begin{cases}
E\\
T\\
q(s)\\
Stage
\end{cases}
\]

通过阶段识别作为 auxiliary loss，帮助模型学习更符合计算机执行机理的隐空间，但阶段识别错误不直接向功率模型硬传播。

---

## 14.4 模型可解释性分析

可以研究：

- 功率峰值对应何种执行阶段；
- 功率谷值是否对应通信或同步；
- 不同并行策略如何改变阶段占比；
- compute-bound 与 communication-bound 任务的 Shape 是否存在稳定差异。

---

# 15. 建议的整体模型结构

输入：

\[
X=
[
X_{\mathrm{task}},
X_{\mathrm{architecture}},
X_{\mathrm{execution}},
X_{\mathrm{hardware}}
]
\]

构造物理派生特征：

\[
\Phi(X)
\]

形成统一输入：

\[
X^{*}=[X,\Phi(X)]
\]

经过共享编码器：

\[
z=\mathrm{Encoder}(X^{*})
\]

设置三个预测分支：

## Energy Head

\[
\hat E=f_E(z)
\]

## Duration Head

\[
\hat T=f_T(z)
\]

## Shape Head

若采用低维基展开：

\[
[\hat a_1,\dots,\hat a_K]=f_q(z)
\]

再由：

\[
\hat q(s)
=
\bar q(s)
+
\sum_{k=1}^{K}\hat a_k\phi_k(s)
\]

重构归一化功率曲线。

最终：

\[
\boxed{
\hat P(t)
=
\frac{\hat E}{\hat T}
\hat q\left(\frac{t}{\hat T}\right)
}
\]

---

# 16. 建议加入的基本物理约束

功率曲线应满足：

## 非负约束

\[
q(s)\geq0
\]

以及：

\[
P(t)\geq0
\]

## 归一化能量约束

\[
\int_0^1 q(s)\,ds=1
\]

从而保证：

\[
\int_0^T P(t)\,dt=E
\]

该约束是 EDS 模型的重要物理一致性基础。

---

# 17. 当前阶段建议形成的研究主线

目前建议将整个课题划分为以下层次。

## 第一阶段：任务到总能耗与持续时间

重点解决：

\[
X+\Phi(X)\rightarrow E
\]

以及：

\[
X+\Phi(X)\rightarrow T
\]

目标是首先建立稳定、可解释、具有一定跨模型和跨硬件泛化能力的宏观映射。

---

## 第二阶段：任务到归一化功率曲线

重点解决：

\[
X+\Phi(X)\rightarrow q(s)
\]

研究：

- 功率曲线形态；
- 周期性；
- 峰谷特性；
- 不同任务类型间差异；
- 不同并行策略导致的形态变化；
- 不同硬件平台下 Shape 的可迁移性。

---

## 第三阶段：功率曲线重构

利用：

\[
\hat E,\hat T,\hat q(s)
\]

重构：

\[
\hat P(t)
=
\frac{\hat E}{\hat T}
\hat q\left(\frac{t}{\hat T}\right)
\]

并验证：

- 能量误差；
- 平均功率误差；
- 峰值误差；
- ramp 误差；
- 时序相似性；
- 高频波动；
- 周期结构。

---

# 18. 当前阶段的重要科学认识

目前可形成以下阶段性判断。

### 认识一

AI 算力任务到功率曲线之间存在直接映射：

\[
X\rightarrow P(t)
\]

无需人为规定必须经过多个独立预测中间层。

### 认识二

显式串联：

\[
Task
\rightarrow Stage
\rightarrow Resource
\rightarrow Power
\]

虽然解释性强，但可能导致误差逐层传播，因此不宜作为当前首选主模型。

### 认识三

执行阶段、计算量、显存访问和通信仍然是极其重要的信息，但更适合作为：

- 物理派生特征；
- 结构先验；
- 辅助任务；
- 解释变量。

### 认识四

相比直接预测变长功率序列，将任务功率曲线分解为：

\[
\boxed{
E,\quad T,\quad q(s)
}
\]

具有更加清晰的物理意义和更好的建模可操作性。

### 认识五

EDS 分解不是认为三者彼此独立，而是在输出空间中分解：

- 能量尺度；
- 时间尺度；
- 功率形态尺度。

### 认识六

真正决定模型跨任务和跨硬件泛化能力的关键，可能不是单纯增加机器学习模型复杂度，而是能否由任务原始配置进一步构造：

\[
FLOPs,\quad
Memory\ Traffic,\quad
Communication,\quad
Arithmetic\ Intensity
\]

等具有物理意义的中间特征。

### 认识七

当前阶段最值得深入研究的问题并不是“采用 LSTM 还是 Transformer”，而是：

> **如何科学定义任务输入特征，并建立任务配置与计算、访存、通信等物理工作量之间的映射。**

这将直接决定后续总能耗、持续时间和 Shape 模型的科学性与泛化能力。

---

# 19. 后续需要重点讨论的问题

后续研究建议依次解决以下问题：

1. 明确“任务提交前可获得变量”和“运行后才能观测变量”的边界；
2. 建立完整的 Task / Architecture / Execution / Hardware 特征字典；
3. 推导不同任务类型对应的 FLOPs、memory traffic 和 communication 物理特征；
4. 判断哪些派生特征能够跨模型架构统一表达；
5. 判断哪些特征只能针对 Training、LLM inference、Diffusion 等分别定义；
6. 建立总能耗 \(E\) 的第一版模型；
7. 建立任务持续时间 \(T\) 的第一版模型；
8. 对现有高频功率数据进行归一化并研究 Shape 的低维结构；
9. 判断 PCA / Functional PCA / Autoencoder 等哪种 Shape 表征最合适；
10. 研究是否需要将执行阶段识别作为辅助任务加入模型；
11. 建立跨模型、跨配置、跨 GPU 的泛化测试方案；
12. 最终将 EDS 模型扩展至节点、集群和数据中心级功率合成。

---

# 20. 当前阶段的核心公式

本阶段建议将以下公式作为课题后续建模工作的核心表达：

\[
\boxed{
X=
[
X_{\mathrm{task}},
X_{\mathrm{architecture}},
X_{\mathrm{execution}},
X_{\mathrm{hardware}}
]
}
\]

\[
\boxed{
X^{*}
=
[X,\Phi(X)]
}
\]

\[
\boxed{
\hat E=f_E(X^{*})
}
\]

\[
\boxed{
\hat T=f_T(X^{*})
}
\]

\[
\boxed{
\hat q(s)=f_q(X^{*})
}
\]

以及最终：

\[
\boxed{
\hat P(t)
=
\frac{\hat E}{\hat T}
\hat q\left(\frac{t}{\hat T}\right)
}
\]

其中：

\[
\int_0^1\hat q(s)\,ds=1
\]

从而保证：

\[
\int_0^{\hat T}\hat P(t)\,dt=\hat E
\]

该框架将“任务到功率”的复杂映射分解为：

> **任务工作量决定能耗尺度，软硬件执行效率决定时间尺度，计算—访存—通信与阶段演化共同决定功率形态。**

这可以作为现阶段课题后续数据分析、模型构建、实验设计和论文研究的统一理论框架。
