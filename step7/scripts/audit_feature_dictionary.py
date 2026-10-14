"""Supplemental per-feature semantics; never modifies frozen model inputs/protocol."""
import json

import pandas as pd

from step7.scripts.common import ROOT,STEP7,read_json


def definition(column):
    base={"source":"step6/analysis/phase4_power_modeling_table.parquet :: "+column,
          "unit":"dimensionless","formula":"copy frozen column without target-dependent transformation",
          "applicability":"all architectures unless frozen static field is unknown",
          "exante_reason":"model/config/hardware/planned-request metadata known before inference",
          "limitation":"observational benchmark metadata; no runtime telemetry"}
    def finish(**updates):
        return {**base,**updates}
    if column=="diag_architecture_class":
        return finish(unit="category",limitation="architecture class only, never model identity or lineage")
    if column=="x_protocol_task":
        return finish(unit="category",applicability="task sensitivity only",limitation="domain/template confounding; excluded from primary tiers")
    if column.startswith("x_deployment_"):
        units={"data_parallel":"replicas","tensor_parallel":"GPUs/TP-group","expert_parallel":"configured EP value","max_num_seqs":"configured concurrent sequences"}
        suffix=column.removeprefix("x_deployment_")
        if suffix in units:
            return finish(unit=units[suffix],limitation="frozen config; EP scope and max_num_seqs per-replica semantics not inferred")
    if column.startswith("x_hardware_"):
        suffix=column.removeprefix("x_hardware_")
        simple={"gpu_model":("category","public GPU label"),"num_gpus":("GPUs","configured allocated card count"),"rated_power_w":("W/GPU","reference rated power"),"aggregate_rated_power_w":("W","num_gpus * rated_power_w"),"hbm_capacity_gb":("GB/GPU (decimal)","reference HBM capacity"),"hbm_bandwidth_gbps":("GB/s/GPU (decimal)","reference HBM peak bandwidth"),"interconnect_bandwidth_gbps":("GB/s/GPU (bidirectional)","reference NVLink bandwidth"),"precision_matched_peak_compute_tflops":("TFLOP/s/GPU (dense)","precision profile lookup; unavailable H100 MXFP4 uses inherited BF16 fallback")}
        if suffix in simple:
            unit,formula=simple[suffix]
            return finish(unit=unit,formula=formula,source=base["source"]+"; step5/config/gpu_hardware_specs.json and hardware_source_manifest.json; step6/scripts/build_modeling_table.py",limitation="H100/B200 ambiguous reference SKU; maxima not run-time power cap or achieved throughput; H100 MXFP4 BF16 fallback is not native FP4 capability")
        old={"physics_compute_time_proxy_seconds":("seconds/request","old prefill_total_FLOPs / (peak_TFLOPs * 1e12 * model_parallel_GPUs)"),"physics_memory_time_proxy_seconds":("seconds/request","(old ideal weight bytes/GPU + old mean KV bytes/GPU) / (HBM_GBps * 1e9)"),"physics_weight_capacity_ratio":("dimensionless","old ideal weight bytes/GPU / (HBM_GB * 1e9)"),"physics_kv_capacity_ratio_mean":("dimensionless","old mean KV bytes/GPU / (HBM_GB * 1e9)"),"physics_compute_to_memory_pressure_ratio":("dimensionless","old compute time proxy / old memory time proxy")}
        if suffix in old:
            unit,formula=old[suffix]
            return finish(unit=unit,formula=formula,source=base["source"]+"; step5/scripts/build_modeling_table.py",applicability="old-B5 sensitivity on compact supported cohort only",limitation="legacy theoretical lower-bound ratios; not measured duration/bandwidth/utilization")
    if column.startswith("x_workload_") or column.startswith("x_text_"):
        if column=="x_workload_max_output_tokens":
            return finish(unit="tokens/request upper bound",formula="configured max_output_tokens",limitation="generation ceiling, not actual output length")
        text=column.startswith("x_text_")
        stat=column.rsplit("_",1)[1]
        if column.endswith("second_moment"):
            stat="second_moment"
        formulas={"mean":"mean(z)","p50":"quantile(z,0.50)","p90":"quantile(z,0.90)","p95":"quantile(z,0.95)","min":"min(z)","max":"max(z)","std":"population std(z), ddof=0","second_moment":"mean(z^2)","rms":"sqrt(mean(z^2))","cv":"population std(z)/mean(z)"}
        if stat in formulas:
            unit="characters" if "characters" in column else "UTF-8 bytes" if text else "tokens"
            unit="dimensionless" if stat=="cv" else unit+("^2/request" if stat=="second_moment" else "/request")
            return finish(unit=unit,formula=formulas[stat]+("; z=sum raw input-part lengths per request, no completion/template markers/tokenizer" if text else "; z=planned per-request input length under frozen benchmark/effective tokenization convention"),source="step7/scripts/build_features.py request_text_audit; frozen request hashes in tokenizer_free_text_audit.csv" if text else base["source"]+"; step3/analysis/"+("effective_workload_provenance.json" if "effective" in column else "planned_workload_provenance.json"),applicability="all three hash-verified raw request tasks; L1T sensitivity" if text else "required planned lengths reliably available; strict tier support",limitation="raw Unicode code-point/UTF8 statistics; template/domain proxy, not model tokens" if text else "benchmark/effective differ by frozen preprocessing; no actual inference outputs")
    if column.startswith("x_model_"):
        suffix=column.removeprefix("x_model_")
        if suffix.startswith("benchmark_"):
            unit="billion parameters" if "params_billions" in suffix else "category"
            return finish(unit=unit,source=base["source"]+"; frozen benchmark model metadata",limitation="benchmark parameter/precision convention, not inferred weight tensors")
        known={"attention_variant":"category","mlp_structure":"category","context_length":"tokens","dense_layer_count":"layers","moe_layer_count":"layers","num_layers":"layers","hidden_size":"hidden units","intermediate_size":"FFN units","head_dim":"units/head","num_attention_heads":"query heads","num_key_value_heads":"KV heads"}
        if suffix in known:
            return finish(unit=known[suffix],source=base["source"]+"; step3/analysis/model_architecture_features.parquet",applicability="frozen detailed static available; hybrid dense/MoE composition structural N/A encoded 0+flag",limitation="Llama static unknown not fabricated; architecture-specific semantics retained")
    if column.startswith("x_p5_extended_"):
        suffix=column.removeprefix("x_p5_extended_")
        flag="applicable" if suffix.endswith("_applicable") else "unknown" if suffix.endswith("_unknown") else None
        raw=suffix.removesuffix("_"+flag) if flag else suffix
        applicable="hybrid only" if "mamba" in raw or "hybrid" in raw else "mixture_of_experts only" if "expert" in raw or "moe" in raw else "MLA only" if "qk_" in raw or "v_head_dim" in raw else "detailed static cohort; sliding_window absent/unknown as frozen"
        unit="binary 0/1" if flag else "tokens" if raw=="sliding_window" else "layers" if "layer_count" in raw else "experts" if "expert_count" in raw else "units" 
        formula="architecture applicability indicator" if flag=="applicable" else "applicable AND original static value missing" if flag=="unknown" else f"x_model_{raw} where applicable, else structural zero; applicable unknown stays NaN"
        return finish(unit=unit,formula=formula,source="step6 frozen x_model_"+raw+"; step7/scripts/build_features.py architecture-specific encoding",applicability=applicable,limitation="L2X sensitivity only; structural zero requires applicability flag; unknown is distinct")
    if column.startswith("x_p5_"):
        raw=column.removeprefix("x_p5_")
        mapping={"precision_bits":("bits/parameter","bfloat16=16, fp8=8, mxfp4=4"),"weight_bytes_total":("bytes/model","total_params_billions * 1e9 * precision_bits / 8"),"active_weight_bytes_total":("bytes/active-model","activated_params_billions * 1e9 * precision_bits / 8"),"active_param_ratio":("dimensionless","activated_params_billions / total_params_billions"),"model_parallel_gpus_per_replica":("GPUs/replica","num_gpus / data_parallel; positive integer checked"),"max_num_seqs_per_gpu":("configured sequences/GPU","max_num_seqs / num_gpus; scale proxy only"),"weight_bytes_per_model_parallel_gpu":("bytes/GPU","weight_bytes_total / model_parallel_gpus_per_replica"),"active_weight_bytes_per_model_parallel_gpu":("bytes/GPU","active_weight_bytes_total / model_parallel_gpus_per_replica"),"weight_hbm_ratio":("dimensionless","weight_bytes_total / model_parallel_gpus_per_replica / (HBM_GB*1e9)"),"active_weight_hbm_ratio":("dimensionless","active_weight_bytes_total / model_parallel_gpus_per_replica / (HBM_GB*1e9)"),"ffn_expansion_ratio":("dimensionless","intermediate_size / hidden_size"),"gqa_ratio":("dimensionless","query_heads / kv_heads for GQA; structural 0 otherwise"),"gqa_ratio_applicable":("binary 0/1","attention_variant == gqa"),"transformer_composition_applicable":("binary 0/1","architecture != hybrid_mamba_transformer"),"moe_layer_fraction":("dimensionless","moe_layer_count / num_layers"),"dense_layer_fraction":("dimensionless","dense_layer_count / num_layers"),"params_per_layer":("billion parameters/layer","total_params_billions / num_layers"),"active_params_per_layer":("billion parameters/layer","activated_params_billions / num_layers")}
        physics={"prefill_projection_flop_per_request":("FLOPs/request","4*N*H*D*(A+K)*mu"),"prefill_attention_flop_per_request":("FLOPs/request","4*N*A*D*E[L^2]"),"prefill_mlp_flop_per_request":("FLOPs/request","2*g*N*H*I*E*mu"),"decode_one_token_flop_mean_context":("FLOPs/generated token (representative)","4*N*H*D*(A+K) + 2*g*N*H*I*E + 4*N*A*D*(mu+1)"),"kv_mean_request_bytes":("bytes/request","2*N*K*D*(runtime_dtype_bits/8)*mu"),"kv_p95_request_bytes":("bytes/request","2*N*K*D*(runtime_dtype_bits/8)*p95(L)"),"kv_max_request_bytes":("bytes/request","2*N*K*D*(runtime_dtype_bits/8)*max(L)")}
        if raw in physics:
            unit,formula=physics[raw]
            return finish(unit=unit,formula=formula,source="step7/scripts/build_features.py compact_physics; physics_feature_formulas_phase5.md",applicability="GQA Qwen/Gemma with complete static/benchmark lengths and known runtime KV dtype",limitation="N layers,H hidden,A heads,K KV heads,D head-dim,I FFN/expert width,E active experts,g linear-term count; full-square attention convention; no MLA/hybrid or shared-expert correction")
        if raw in mapping:
            unit,formula=mapping[raw]
            return finish(unit=unit,formula=formula,source="step7/scripts/build_features.py using frozen model/config/hardware metadata",limitation="deterministic static proxy; ideal residency excludes uneven sharding/EP sharing and quantization metadata overhead; architecture flags distinguish N/A")
    raise ValueError("undefined or forbidden feature: "+column)


def main():
    table=pd.read_parquet(STEP7/"analysis/phase5_master_table.parquet")
    levels=read_json(STEP7/"config/feature_hierarchy_phase5.json")["levels"]
    amended=pd.read_parquet(STEP7/"analysis/phase5_l2x_amended_table.parquet") if (STEP7/"config/l2x_semantic_amendment.json").exists() else None
    records=[]
    for column in sorted({c for b in levels.values() for c in b["columns"]}):
        applicability=table[column+"_applicable"].astype(bool) if column+"_applicable" in table else (~table.diag_architecture_class.eq("hybrid_mamba_transformer") if column in {"x_model_dense_layer_count","x_model_moe_layer_count","x_p5_dense_layer_fraction","x_p5_moe_layer_fraction"} else pd.Series(True,index=table.index))
        for name,bundle in levels.items():
            if column not in bundle["columns"]:
                continue
            effective=amended if amended is not None and name=="L2X-StaticExtended" else table
            subset=effective.loc[effective[bundle["eligibility"]]]
            level_applicable=effective[column+"_applicable"].astype(bool) if column+"_applicable" in effective else applicability
            records.append({"feature":column,"feature_level":name,**definition(column),"effective_table":"phase5_l2x_amended_table.parquet" if effective is amended else "phase5_master_table.parquet (original freeze)","support_rows":len(subset),"support_models":subset.diag_model_id.nunique(),"missing_rows":int(subset[column].isna().sum()),"unique_count":int(subset[column].nunique()),"structural_na_rows":int((~level_applicable.loc[subset.index]).sum()),"unknown_rows":int((level_applicable.loc[subset.index]&subset[column].isna()).sum())})
    audit=pd.DataFrame(records)
    audit.to_csv(STEP7/"analysis/feature_dictionary_phase5.csv",index=False)
    if amended is not None:
        import shutil
        for filename in ["feature_semantics_audit.csv","feature_coverage.csv"]:
            path=STEP7/"analysis"/filename
            archived=STEP7/"analysis/superseded_l2x"/filename
            if not archived.exists(): shutil.copy2(path,archived)
            aggregate=pd.read_csv(path)
            for column in read_json(STEP7/"config/l2x_semantic_amendment.json")["changed_columns"]:
                applicable=amended[column+"_applicable"].astype(bool) if column+"_applicable" in amended else pd.Series(True,index=amended.index)
                mask=aggregate.feature.eq(column)
                values={"source":"Phase5 frozen semantic amendment; phase5_l2x_amended_table.parquet","missing_rows":int(amended[column].isna().sum()),"unique_count":int(amended[column].nunique()),"structural_na_rows":int((~applicable).sum()),"unknown_rows":int((applicable&amended[column].isna()).sum())}
                for key,value in values.items(): aggregate.loc[mask,key]=value
            aggregate.to_csv(path,index=False)
    (STEP7/"reports/feature_dictionary_phase5.md").write_text("# Phase5 逐字段语义字典\n\n完整逐feature×level记录见 analysis/feature_dictionary_phase5.csv：来源、单位、公式、执行前可知理由、适用架构、局限、各层覆盖、缺失、unique、结构N/A和unknown计数。此文件补充说明冻结输入，不改动protocol/master或研究候选。\n\n共享符号和公式约定见 physics_feature_formulas_phase5.md；HF请求哈希见 tokenizer_free_text_audit.csv。单位GB/GBps使用十进制。benchmark precision仅表示权重格式，不自动等于KV运行dtype。\n",encoding="utf-8")
    print(json.dumps({"features":audit.feature.nunique(),"feature_level_rows":len(audit)}))


if __name__=="__main__":
    main()
