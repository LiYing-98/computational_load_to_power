"""Build source-backed Phase5 scientific figures, report and read-only notebook."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import numpy as np
import pandas as pd

from step7.scripts.common import ROOT,STEP7,TARGET,read_json,write_json

MAIN_LEVELS=["L0-Basic","L0A-ActiveAware","L1-Deployment","L2-StaticCore","L3-Workload","L4-CompactPhysics"]
COLORS={"T-total":"#176B87","G-perGPU":"#B06B18"}
REQUIRED_FIGURES=["information_accuracy_curve.png","algorithm_feature_heatmap.png","total_per_gpu_paired.png","gpu_count_error_scaling.png","calibration_overall_8gpu_top10.png","feature_increment_forest.png"]


def require_complete_matrix(status):
    if status["status"]!="complete" or status["completed_experiments"]!=status["expected_experiments"]:
        raise ValueError("partial matrix cannot be presented as final research")


def md_table(frame,columns):
    data=frame[columns].copy()
    lines=["| "+" | ".join(columns)+" |","|"+"|".join("---" for _ in columns)+"|"]
    for values in data.itertuples(index=False,name=None):
        lines.append("| "+" | ".join(f"{v:.3f}" if isinstance(v,(float,np.floating)) else str(v) for v in values)+" |")
    return "\n".join(lines)


def save_figure(fig,name):
    fig.tight_layout()
    fig.savefig(STEP7/"figures"/name,dpi=160,bbox_inches="tight")
    plt.close(fig)


def selected_oof(level,lock):
    choice=lock["choices"][level]
    return pd.read_parquet(STEP7/"analysis"/f"oof_predictions_{choice['experiment_id']}.parquet")


def draw_cluster_intervals(ax,point,y,low,high,*,color,marker,label):
    """Draw percentile endpoints directly: the interval need not contain the point."""
    ax.hlines(y,low,high,color=color)
    ax.plot(point,y,marker=marker,linestyle="none",color=color,label=label)


def build_figures(matrix,increments,slice_table,lock):
    directory=STEP7/"figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":"#d0d5dd","grid.alpha":.5})
    main=matrix.loc[matrix.analysis_role.eq("main")]
    winners=main.sort_values(["mae","rmse"]).groupby(["feature_level","target_strategy"],sort=False).head(1)
    winners.to_csv(STEP7/"reports/information_curve_source.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,5))
    for strategy in COLORS:
        data=winners.loc[winners.target_strategy.eq(strategy)].set_index("feature_level").reindex(MAIN_LEVELS)
        ax.plot(np.arange(6),data.mae,marker="o" if strategy=="T-total" else "s",color=COLORS[strategy],label=strategy)
    counts=main.groupby("feature_level").n_rows.first()
    ax.set_xticks(np.arange(6),[f"{x.split('-')[0]}\nn={counts[x]}" for x in MAIN_LEVELS])
    ax.set(xlabel="Information level (each tier's own support)",ylabel="Observed best OOF total MAE (W)",title="Information and observed engineering accuracy")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.text(.05,.01,"Supports differ above L1; this curve does not estimate incremental feature gains. Winner selection bias applies.",fontsize=9)
    save_figure(fig,REQUIRED_FIGURES[0])
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for strategy,ax in zip(COLORS,axes):
        pivot=main.loc[main.target_strategy.eq(strategy)].pivot(index="algorithm",columns="feature_level",values="mae").reindex(columns=MAIN_LEVELS)
        image=ax.imshow(pivot.to_numpy(),cmap="Blues",norm=LogNorm(vmin=max(1,main.mae.min()),vmax=main.mae.max()),aspect="auto")
        ax.set_xticks(np.arange(6),[x.split('-')[0] for x in MAIN_LEVELS])
        ax.set_yticks(np.arange(len(pivot)),pivot.index)
        ax.set_title(strategy+" (W; log color scale)")
        ax.grid(False)
        for i in range(len(pivot)):
            for j in range(6):
                value=pivot.iloc[i,j]
                ink="white" if image.norm(value)>.62 else "#101828"
                ax.text(j,i,f"{value:.0f}",ha="center",va="center",fontsize=9,color=ink)
        fig.colorbar(image,ax=ax,label="Total MAE (W)")
    save_figure(fig,REQUIRED_FIGURES[1])
    total=main.loc[main.target_strategy.eq("T-total"),["feature_level","algorithm","cohort","mae"]].rename(columns={"mae":"T_total_mae"})
    per=main.loc[main.target_strategy.eq("G-perGPU"),["feature_level","algorithm","cohort","mae"]].rename(columns={"mae":"G_restored_total_mae"})
    paired=total.merge(per,on=["feature_level","algorithm","cohort"],validate="one_to_one")
    paired.to_csv(STEP7/"reports/target_paired_plot_source.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,6))
    markers=["o","s","^","D","v","P"]
    for level,marker in zip(MAIN_LEVELS,markers):
        rows=paired.loc[paired.feature_level.eq(level)]
        ax.scatter(rows.T_total_mae,rows.G_restored_total_mae,marker=marker,facecolors="none",edgecolors="#176B87",label=level.split('-')[0],s=65)
    low=0; high=max(paired.T_total_mae.max(),paired.G_restored_total_mae.max())*1.05
    ax.plot([low,high],[low,high],"--",color="#555555",label="Equal total MAE")
    ax.set(xlim=(low,high),ylim=(low,high),xlabel="T-total MAE (W)",ylabel="G-perGPU restored total MAE (W)",title="Paired target representation")
    ax.set_aspect("equal")
    ax.legend(frameon=False,fontsize=9)
    save_figure(fig,REQUIRED_FIGURES[2])
    choice=lock["choices"]["L1-Deployment"]
    scale=slice_table.loc[slice_table.feature_level.eq("L1-Deployment")&slice_table.cohort.eq("full")&slice_table.algorithm.eq(choice["algorithm"])&slice_table.slice_type.eq("gpu_count")].copy()
    scale.to_csv(STEP7/"reports/gpu_count_scaling_source.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for strategy in COLORS:
        rows=scale.loc[scale.target_strategy.eq(strategy)].copy()
        rows["gpu_count"]=rows.slice_value.astype(int)
        rows=rows.sort_values("gpu_count")
        for ax,column,label in [(axes[0],"mae","Total MAE (W)"),(axes[1],"per_gpu_mae","Average per-GPU MAE (W/GPU)")]:
            ax.plot(rows.gpu_count,rows[column],marker="o" if strategy=="T-total" else "s",color=COLORS[strategy],label=strategy)
            ax.set(xlabel="GPU count",ylabel=label,title=choice["algorithm"]+" at L1")
            ax.set_xticks([1,2,4,8]); ax.set_ylim(bottom=0)
            ax.legend(frameon=False)
    save_figure(fig,REQUIRED_FIGURES[3])
    oof=selected_oof("L1-Deployment",lock)
    subsets=[("Overall",oof),("8-GPU",oof.loc[oof.x_hardware_num_gpus.eq(8)]),("Top 10% total power",oof.loc[oof[TARGET]>=oof[TARGET].quantile(.9)])]
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    source=[]
    for (name,rows),ax in zip(subsets,axes):
        high=max(rows[TARGET].max(),rows.prediction.max())*1.03; low=min(0,rows.prediction.min())
        slope,intercept=np.polyfit(rows[TARGET],rows.prediction,1)
        ax.scatter(rows[TARGET],rows.prediction,s=15,alpha=.6,color=COLORS[choice["target_strategy"]])
        ax.plot([low,high],[low,high],"--",color="#555555")
        ax.set(xlabel="Measured total W",ylabel="Predicted total W",title=f"{name}, n={len(rows)}\nslope={slope:.3f}, intercept={intercept:.1f}W",xlim=(low,high),ylim=(low,high))
        ax.set_aspect("equal")
        source.append(rows.assign(calibration_subset=name))
    pd.concat(source).to_parquet(STEP7/"analysis/calibration_figure_source.parquet",index=False)
    save_figure(fig,REQUIRED_FIGURES[4])
    primary_pairs=[("L0-Basic","L0A-ActiveAware"),("L0A-ActiveAware","L1-Deployment"),("L1-Deployment","L2-StaticCore"),("L2-StaticCore","L3-Workload"),("L3-Workload","L4-CompactPhysics")]
    fig,axes=plt.subplots(3,2,figsize=(15,15))
    for (lower,higher),ax in zip(primary_pairs,axes.flat):
        subset=increments.loc[increments.lower.eq(lower)&increments.higher.eq(higher)]
        algorithms=sorted(subset.algorithm.unique())
        for strategy,offset in [("T-total",-.15),("G-perGPU",.15)]:
            rows=subset.loc[subset.target_strategy.eq(strategy)].set_index("algorithm").reindex(algorithms)
            center=rows.row_weighted_delta_mae.to_numpy()
            draw_cluster_intervals(ax,center,np.arange(len(algorithms))+offset,rows.row_ci_low.to_numpy(),rows.row_ci_high.to_numpy(),marker="o" if strategy=="T-total" else "s",color=COLORS[strategy],label=strategy)
        ax.set_yticks(np.arange(len(algorithms)),algorithms)
        ax.axvline(0,color="#555555",linestyle="--")
        ax.set(xlabel="Higher − lower total MAE (W), 95% model-cluster CI",title=lower.split('-')[0]+" → "+higher.split('-')[0])
        ax.legend(frameon=False,fontsize=9)
    axes.flat[-1].axis("off")
    save_figure(fig,REQUIRED_FIGURES[5])
    write_json(STEP7/"reports/figure_sources.json",{REQUIRED_FIGURES[0]:"information_curve_source.csv (own supports; winner bias)",REQUIRED_FIGURES[1]:"main_model_matrix.csv main rows",REQUIRED_FIGURES[2]:"target_paired_plot_source.csv",REQUIRED_FIGURES[3]:"gpu_count_scaling_source.csv",REQUIRED_FIGURES[4]:"analysis/calibration_figure_source.parquet",REQUIRED_FIGURES[5]:"feature_increment_effects.csv matched-support row-weighted intervals"})
    return winners


def effect_statement(effects,lower,higher):
    rows=effects.loc[effects.lower.eq(lower)&effects.higher.eq(higher)]
    if rows.empty:
        return "无可比较的严格支持集；需保留coverage boundary。"
    beneficial=rows.loc[rows.row_ci_high.lt(0)]
    harmful=rows.loc[rows.row_ci_low.gt(0)]
    return f"在相同支持集（{int(rows.n_rows.iloc[0])}行）上，{len(rows)}组 algorithm×target 比较的ΔMAE范围为 {rows.row_weighted_delta_mae.min():.1f} 至 {rows.row_weighted_delta_mae.max():.1f} W；{len(beneficial)}组95%模型簇区间全部低于0，{len(harmful)}组全部高于0，其余跨0。完整明细包括row-weighted和equal-model口径，见 feature_increment_effects.csv。"


def build_report(matrix,effects,targets,slice_table,lock,winners):
    coverage=pd.read_csv(STEP7/"analysis/cohort_coverage.csv")
    stresses=pd.read_csv(STEP7/"reports/generalization_stress.csv")
    main=matrix.loc[matrix.analysis_role.eq("main")]
    best_l0=main.loc[main.feature_level.eq("L0-Basic")].sort_values("mae").iloc[0]
    l1choice=lock["choices"]["L1-Deployment"]
    low_slice=slice_table.loc[slice_table.experiment_id.eq(l1choice["experiment_id"])&slice_table.slice_type.eq("gpu_count")]
    targetmain=targets.merge(main[["feature_level","cohort"]].drop_duplicates(),on=["feature_level","cohort"],validate="many_to_one")
    improvements=targetmain.loc[targetmain.row_ci_high.lt(0)]
    negative=int(matrix.negative_predictions.sum())
    text=f"""# Phase 5：事前信息完整度与总功率/平均单GPU目标的预测比较

## 执行摘要

所有主矩阵、相同支持集重跑和敏感性单元共{len(matrix)}组，均采用5-fold model-group outer CV、3-fold model-group inner CV，算法独立调参。最低信息L0的最低观察OOF MAE为 **{best_l0.mae:.1f} W**（{best_l0.algorithm}、{best_l0.target_strategy}），per-GPU MAE为 **{best_l0.per_gpu_mae:.1f} W/GPU**。这是多个候选中最低观察值，有winner-selection bias，不是新的独立最终性能。

原565条stable records、27 models完整保留；详细结构和workload支持集按真实字段重算，physics和effective的缺失边界保留。既有Phase4 Confirmatory已经被查看，本阶段所有结果均是nested OOF或明确标记的post-hoc诊断。正式对象仍为GPU侧、文本LLM inference、stable、high-load / near-saturation steady-state。

## RQ1：最低信息模型的能力

L0只要求GPU型号/卡数、总参数量、架构类别和benchmark precision；其硬件lookup及weight bytes自动派生不增加额外用户输入负担。L0/L0A/L1主支持集均565条。按相同L0支持集的算法独立nested OOF：最低MAE {best_l0.mae:.1f} W，RMSE {best_l0.rmse:.1f} W，R² {best_l0.r2:.3f}，MdAPE {best_l0.mdape:.2f}%，sMAPE {best_l0.smape:.2f}%。各算法独立调参，inner CV不混合竞争；本阶段每个非基线算法统一8候选，较默认24候选缩减，搜索覆盖有限。

## RQ2：active params 与部署信息的独立价值

L0→L0A：{effect_statement(effects,'L0-Basic','L0A-ActiveAware')}

L0A→L1：{effect_statement(effects,'L0A-ActiveAware','L1-Deployment')}

这些比较均565-row support、完全相同run keys/outer folds，不能把多个学习器结果平均成一个普适“物理增益”。model_parallel=num_gpus/DP先校验整数；weight residency是理想分配代理，不描述EP跨DP共享expert的实际显存分布。max_num_seqs_per_gpu仅为明确的尺度代理，未假设per-replica配置scope。

## RQ3：详细static结构的matched-support增益

{effect_statement(effects,'L1-Deployment','L2-StaticCore')}

L1@L2-support已重新训练；没有拿L1@565与L2较小cohort的绝对MAE相减。Hybrid的dense/MoE transformer layer composition为结构N/A并带applicability标记，特殊layer pattern仅进入L2X；unknown仍缺失并有indicator。Llama未伪造层数、KV heads或tokenizer。L2X扩展的增量同样在严格共同支持集判断。

## RQ4：workload独立增量与benchmark/effective

L2→L3：{effect_statement(effects,'L2-StaticCore','L3-Workload')}

Benchmark→effective：{effect_statement(effects,'L3-Workload','L3E-Effective')}

后者在effective可靠可用的相同support/folds上比较；effective不自动被解释成更高阶。L3仅使用intensity统计与max_output_tokens，request repeats、总请求和total tokens未进入主层。Tokenizer-free characters/UTF-8 bytes通过全部三类冻结请求哈希验证，覆盖565，无completion或tokenizer；L1T可作为探索代理，但任务/domain模式和模板影响仍可能混杂。

## RQ5：compact physics是否增加独立信息

{effect_statement(effects,'L3-Workload','L4-CompactPhysics')}

精简代理来自benchmark mean/second moment/p95/max与冻结静态结构；MLA/hybrid不套GQA公式，运行KV dtype未知不伪造。总prefill FLOPs是三块确定性加和，保留审计值但从正式X去掉；weight residency已经在L1，不重复添加。full-square attention约定继承旧研究，忽略sliding window和真实kernel细节。physics支持仅Qwen/Gemma，不能推断跨所有谱系稳定；旧B5比率仅sensitivity。公式与边界见physics_feature_formulas_phase5.md。

L4为L3已有事前信息的确定性变换，不新增独立观测信息；它检验的是表示方式对有限学习器的价值，并非新增物理测量或因果机制。当前多数算法×target区间跨0，少数得到支持，不能称为跨算法稳定的普适收益。

## RQ6：G-perGPU是否改善卡数尺度稳定性

同算法/level/cohort的paired comparison中，ΔMAE定义为G恢复total后减T-total；主层相关比较的点估计范围为 {targetmain.row_weighted_delta_mae.min():.1f} 至 {targetmain.row_weighted_delta_mae.max():.1f} W，{len(improvements)}组的模型簇95%CI全部低于0。需逐algorithm×level解释，不能从低per-GPU误差自动推断total更准。G的inner和final-group CV都以恢复后total MAE选参数；num_gpus继续在X中。

L1工程候选的卡数切片（{l1choice['algorithm']} / {l1choice['target_strategy']}）：

{md_table(low_slice,['slice_value','n_rows','mae','per_gpu_mae','signed_bias','per_gpu_signed_bias'])}

同一算法的T/G卡数切片见gpu_count_scaling_source.csv；8卡total误差需要结合W/GPU解释。平均单GPU口径指aggregate/卡数，不是逐卡独立测量。本矩阵负功率预测计数（跨实验重复support，非unique run数）为{negative}，保持原值没有静默clip；逐算法和slice可查prediction原始输出。

修正后负值仅出现在L2X sensitivity的Ridge/ElasticNet T-total基线，各29个，集中于outer fold0的两个Gemma模型；主层及所有G-perGPU单元均无负值。这些线性基线没有非负约束，反映扩展静态组合的外推风险，不是实测负功率。没有据此改参数或删记录；范围、数值和保存OOF来源见`negative_prediction_audit.md`，它们未进入研发候选。

卡数切片显示的是误差幅度/偏差，不能单凭MAE曲线证明条件残差方差已稳定：GPU count同时与GPU型号、模型、并行配置混杂。平均单GPU误差缩小仍须结合恢复total后的paired区间与校准判断，不作为因果消除异方差的证明。

## RQ7：不同算法在哪些信息层适配

各层按自身support最低观察值：

{md_table(winners.sort_values(['feature_level','target_strategy']),['feature_level','target_strategy','algorithm','n_rows','n_models','mae','rmse','r2','per_gpu_mae'])}

Information-Accuracy Curve的高层support不同，只描绘各信息要求下工程可达性能；特征增益以matched-support effects为准。完整8算法×6主层×2目标明细：

{md_table(main.sort_values(['feature_level','algorithm','target_strategy']),['feature_level','algorithm','target_strategy','n_rows','mae','rmse','r2','mdape','smape','signed_bias','per_gpu_mae'])}

## RQ8：分层development candidates及适用边界

L0/L1/L2（并保留L3）候选由各层main nested OOF选择算法和target，在完整对应support上独立3-fold GroupCV从冻结候选列表选最终参数并拟合；metadata设置development_candidate_not_external_validated=true。它们是研发候选，不是外部验证后的通用GPU或node模型。选择规则先锁定，再运行post-hoc stresses。

{md_table(pd.DataFrame(lock['choices'].values()),['feature_level','algorithm','target_strategy','cohort','n_rows','mae','per_gpu_mae'])}

## 泛化压力与校准（post-hoc only）

以下为L0/L1/L2锁定研发候选在leave-one-lineage、within-task-lineage、architecture holdout上的固定参数压力测试，完全使用保存manifest；不回流选择、超参数或feature tiers。最困难的slice如下，全部结果见generalization_stress.csv：

{md_table(stresses.sort_values('mae',ascending=False).head(12),['feature_level','scheme','held_out_group','task','n_rows','n_models','mae','per_gpu_mae','signed_bias'])}

Task-lineage confounding、GPU composition、task composition单独记录。只有H100/B200两个reference points，不能声明跨GPU世代；hybrid支持主要为Nemotron，架构holdout也与谱系混杂。Overall/8GPU/top10校准图及CSV用于判断压缩、异方差和系统偏差，不用于追加调参。

## 敏感性分支的配对证据

L1→tokenizer-free L1T：{effect_statement(effects,'L1-Deployment','L1T-TokenizerFreeTextProxy')}

L2→架构扩展L2X：{effect_statement(effects,'L2-StaticCore','L2X-StaticExtended')}

L4→旧B5：{effect_statement(effects,'L4-CompactPhysics','L4B5-OldB5Sensitivity')}

L1→task label：{effect_statement(effects,'L1-Deployment','L1-TaskSensitivity')}

这些分支不用于改选主研发候选。Tokenizer-free统计在本benchmark中仅有三组任务级长度分布，不能据此宣称跨新workload泛化。

## 独立审查与透明语义修正

审查发现L2X把明确禁用滑动窗口的276条Qwen记录/12模型误当unknown；原config原字节哈希均与冻结Step3 architecture evidence一致，历史文件未改。保留原protocol/master和superseded evidence；在冻结`l2x_semantic_amendment.json`后仅重跑16个L2X单元，原448行、folds、候选、objective不变，208个原始单元保留。`phase5_master_table.parquet`是原预修正快照；L2X正式有效输入为`phase5_l2x_amended_table.parquet`，禁用window=0、applicable=0、unknown=0。上述敏感性证据采用修正后的OOF；候选锁保持原样，source_matrix哈希对应归档原矩阵，而全部main行数值保持不变。

OOF原始/修正指纹已由独立验证器重建，stress cache加入共享建模代码和protocol来源；错误指纹的fault-injection回归必须失败。完整审查发现、修复和审查限制见`final_code_review.md`与`source_integrity_issue.md`。修正是语义缺陷修复，不是看结果后追加特征或扩搜索。

## 数据、口径和可复现性

{md_table(coverage,['feature_level','n_rows','n_models','n_lineages','n_architectures'])}

正式X仅执行前可知；model ID/lineage/subfamily只用于分组/诊断，x_protocol_task仅最后的task sensitivity。所有预处理fold-local，类别未知ignore；真正unknown不能当N/A。Source/protocol/candidate/fold/code/version hashes和checkpoint指纹可复核；ElasticNet弱正则相关输入的数值收敛通过同objective ridge初始化修复，候选空间未改，剩余失败逐项记录，不删失败fold后宣称完整。

## 限制和下一阶段建议

1. 本阶段统一8候选而非默认24，探索预算较小；不可在已看outer结果后偷偷扩搜索并保留原“冻结”声明。
2. Best-of-level最低OOF有winner-selection bias；Phase4 Confirmatory已看，本阶段不产生新的blind confirmatory。
3. N/A编码与理想weight residency/compute proxy是有边界的结构表示；不对应实测utilization、bandwidth或per-card功率。
4. 明确分层L0/L1/L2信息要求，可供后续真实事前预测接口使用；新增外部数据再验证困难谱系、高功率和GPU-generation外推。
5. 相邻信息收益和target收益若CI跨0，结论为当前模型簇证据不足；不能将“未显著”写成物理机制不存在。区间是当前OOF管线下的模型簇重采样区间，多个比较未作family-wise校正，不覆盖跨数据集迁移或全部模型选择不确定性。
6. H100/B200实际SKU未获得逐run确认，硬件额定功率不是配置power cap。继承的H100 MXFP4算力lookup使用BF16参考代理，不代表H100具有原生FP4峰值；不能把该字段当实测或严格precision-matched能力。

六类最低图表均有机器可读来源，notebook只加载persisted evidence并绘图；完整验收见verification_receipt.json。所有step1–step6正式文件保持只读。
"""
    (STEP7/"reports/phase5_hierarchical_modeling_report.md").write_text(text,encoding="utf-8")


def build_notebook():
    choices=read_json(STEP7/"config/development_candidates_lock.json")["choices"]
    summary="; ".join(f"{level.split('-')[0]} observed best total MAE {choices[level]['mae']:.1f} W on {choices[level]['n_rows']} runs ({choices[level]['algorithm']}, {choices[level]['target_strategy']})" for level in ["L0-Basic","L1-Deployment","L2-StaticCore"])
    notebook=nbf.v4.new_notebook()
    notebook.metadata["kernelspec"]={"display_name":"Python Phase5","language":"python","name":"python3"}
    notebook.cells=[
        nbf.v4.new_markdown_cell("# Phase 5 — hierarchical ex-ante GPU power\n\n## tl;dr\n\n"+summary+". These own-support minima are not matched-support feature gains. All 224 cells are complete. Best-of-level has winner selection bias; historical Phase4 confirmatory is not reused as blind evaluation."),
        nbf.v4.new_markdown_cell("## Context & Methods\n\nGPU-side stable high-load text LLM inference only. L0/L0A/L1 preserve565 runs; higher support is based on semantic availability. Eight frozen candidates per nonbaseline learner, five model-group outer folds and three inner folds. Both target strategies select hyperparameters on reconstructed total watts.\n\n### Key Assumptions\n\nHardware profiles are H100/B200 reference proxies. Missing is distinct from structural N/A. Physics excludes unsupported MLA/hybrid and unknown KV dtype. Adjacent effects use matched supports."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import Image,display\nROOT=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'step7/config/protocol_freeze.json').exists()),None)\nif ROOT is None:\n raise FileNotFoundError('Open inside the delivered repository with the step7 evidence files present')\nSTEP7=ROOT/'step7'\nREPORTS=STEP7/'reports'\ncoverage=pd.read_csv(STEP7/'analysis/cohort_coverage.csv')\nmatrix=pd.read_csv(REPORTS/'main_model_matrix.csv')\nincrements=pd.read_csv(REPORTS/'feature_increment_effects.csv')\ntargets=pd.read_csv(REPORTS/'target_representation_comparison.csv')\nslices=pd.read_csv(REPORTS/'slice_metrics.csv')\nlock=json.loads((STEP7/'config/development_candidates_lock.json').read_text())\nassert matrix.experiment_id.nunique()==224\nplt.rcParams.update({'figure.figsize':(10,5),'axes.grid':True,'grid.alpha':.4,'axes.prop_cycle':plt.cycler(color=['#176B87','#B06B18'])})"),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell("display(coverage[['feature_level','n_rows','n_models','n_lineages','n_architectures']])\ndisplay(pd.read_csv(STEP7/'analysis/tokenizer_free_text_audit.csv'))"),
        nbf.v4.new_markdown_cell("## Results\n\n### Information and observed accuracy\n\nSupports above L1 differ; use paired increments rather than this curve for feature-value claims."),
        nbf.v4.new_code_cell("curve=pd.read_csv(REPORTS/'information_curve_source.csv')\nlevels=['L0-Basic','L0A-ActiveAware','L1-Deployment','L2-StaticCore','L3-Workload','L4-CompactPhysics']\nfig,ax=plt.subplots()\nfor strategy,marker in [('T-total','o'),('G-perGPU','s')]:\n data=curve.loc[curve.target_strategy.eq(strategy)].set_index('feature_level').reindex(levels)\n ax.plot(range(6),data.mae,marker=marker,label=strategy)\nax.set_xticks(range(6),[x.split('-')[0] for x in levels])\nax.set_ylabel('Observed best total MAE (W)')\nax.set_ylim(bottom=0)\nax.legend()\nplt.show()\ndisplay(curve[['feature_level','target_strategy','algorithm','n_rows','mae','per_gpu_mae']].round(3))"),
        nbf.v4.new_markdown_cell("### Matched-support feature and target effects"),
        nbf.v4.new_code_cell("display(increments[['lower','higher','algorithm','target_strategy','n_rows','row_weighted_delta_mae','row_ci_low','row_ci_high']].head(24).round(3))\ndisplay(targets[['feature_level','algorithm','n_rows','row_weighted_delta_mae','row_ci_low','row_ci_high']].head(24).round(3))\ndisplay(Image(filename=str(STEP7/'figures/feature_increment_forest.png'),width=1100))\ndisplay(Image(filename=str(STEP7/'figures/total_per_gpu_paired.png'),width=750))"),
        nbf.v4.new_markdown_cell("### GPU-count scaling and calibration\n\nAverage per-GPU error is aggregate error divided by card count, not a per-card measurement."),
        nbf.v4.new_code_cell("display(pd.read_csv(REPORTS/'gpu_count_scaling_source.csv')[['target_strategy','slice_value','n_rows','mae','per_gpu_mae','signed_bias']].round(3))\ndisplay(Image(filename=str(STEP7/'figures/gpu_count_error_scaling.png'),width=1000))\ndisplay(Image(filename=str(STEP7/'figures/calibration_overall_8gpu_top10.png'),width=1200))"),
        nbf.v4.new_markdown_cell("### Development candidates and post-hoc stresses"),
        nbf.v4.new_code_cell("display(pd.DataFrame(lock['choices'].values())[['feature_level','algorithm','target_strategy','n_rows','mae','per_gpu_mae']].round(3))\nstress=pd.read_csv(REPORTS/'generalization_stress.csv')\ndisplay(stress[['feature_level','scheme','held_out_group','task','n_rows','mae','per_gpu_mae','signed_bias']].sort_values('mae',ascending=False).head(12).round(3))"),
        nbf.v4.new_markdown_cell("## Takeaways\n\nUse same-support paired estimates for information gains; compare both reconstructed total and average per-GPU errors for target representation. Candidate choices are engineering development fits without external validation. Stress tests are post-hoc and never feed back into feature/algorithm/hyperparameter decisions. The Chinese scientific report answers RQ1–RQ8 and records coverage, search-budget and winner-selection limitations."),
    ]
    nbf.validate(notebook)
    notebook=NotebookClient(notebook,timeout=600,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute()
    directory=STEP7/"notebooks"; directory.mkdir(exist_ok=True)
    nbf.write(notebook,directory/"phase5_hierarchical_modeling.ipynb")
    exporter=HTMLExporter(template_name="lab")
    html,_=exporter.from_notebook_node(notebook)
    (directory/"phase5_hierarchical_modeling.html").write_text(html,encoding="utf-8")


def main():
    require_complete_matrix(read_json(STEP7/"analysis/matrix_progress.json"))
    if read_json(STEP7/"analysis/statistical_analysis_status.json")["status"]!="complete":
        raise ValueError("paired statistics incomplete")
    matrix=pd.read_csv(STEP7/"reports/main_model_matrix.csv")
    increments=pd.read_csv(STEP7/"reports/feature_increment_effects.csv")
    targets=pd.read_csv(STEP7/"reports/target_representation_comparison.csv")
    slice_table=pd.read_csv(STEP7/"reports/slice_metrics.csv")
    lock=read_json(STEP7/"config/development_candidates_lock.json")
    winners=build_figures(matrix,increments,slice_table,lock)
    build_report(matrix,increments,targets,slice_table,lock,winners)
    build_notebook()
    print("Phase5 six figure families, report and executed notebook built")


if __name__ == "__main__":
    main()
