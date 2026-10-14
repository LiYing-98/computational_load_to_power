"""Independent Phase5 verification; does not import modeling/feature helpers."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import importlib.metadata
import platform
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
STEP7=ROOT/"step7"
TARGET="y_gpu_avg_power_watts"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    path=Path(path)
    if path.suffix.lower() in {".json",".py",".csv",".md",".txt",".ipynb",".html"}:
        content=path.read_text(encoding="utf-8").replace("\r\n","\n").replace("\r","\n").encode("utf-8")
        return hashlib.sha256(content).hexdigest()
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def independent_metrics(actual,prediction,num_gpus):
    actual,prediction,num_gpus=map(lambda a:np.asarray(a,dtype=float),(actual,prediction,num_gpus))
    error=prediction-actual
    magnitude=np.abs(error)
    denominator=np.square(actual-actual.mean()).sum()
    slope,intercept=np.polyfit(actual,prediction,1) if len(actual)>1 and np.ptp(actual)>0 else (np.nan,np.nan)
    return {"mae":float(magnitude.mean()),"rmse":float(np.sqrt(np.square(error).mean())),"r2":float(1-np.square(error).sum()/denominator) if denominator>0 else np.nan,"mdape":float(np.median(magnitude/np.abs(actual))*100),"smape":float(np.mean(200*magnitude/(np.abs(actual)+np.abs(prediction)))),"signed_bias":float(error.mean()),"per_gpu_mae":float((magnitude/num_gpus).mean()),"per_gpu_rmse":float(np.sqrt(np.square(error/num_gpus).mean())),"per_gpu_signed_bias":float((error/num_gpus).mean()),"calibration_slope":float(slope),"calibration_intercept":float(intercept),"negative_predictions":int((prediction<0).sum())}


def need(value,message):
    if not bool(value):
        raise AssertionError(message)


def independent_fingerprint(spec):
    # Reconstruct rather than import the training/checkpoint implementation.
    package_path=ROOT/"step5/.packages"
    if str(package_path) not in sys.path:
        sys.path.insert(0,str(package_path))
    environment={"python":platform.python_version()}
    for p in ["numpy","pandas","scikit-learn","xgboost","catboost"]:
        try: environment[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: environment[p]="unavailable"
    amended=(STEP7/"config/l2x_semantic_amendment.json").exists() and spec["feature_level"]=="L2X-StaticExtended"
    payload={"spec":spec,"protocol":load(STEP7/"config/protocol_freeze.json"),"source_freeze":digest(STEP7/"config/source_freeze_phase5.json"),"versions":environment,"code":{n:digest(STEP7/"scripts"/n) for n in (["amend_l2x.py","modeling.py","common.py","freeze_protocol.py"] if amended else ["run_matrix.py","modeling.py","common.py","freeze_protocol.py"])},"row_order":"run_key ascending; same frozen support and nested groups" if amended else "run_key ascending; fit and hash both canonical"}
    if amended: payload["amendment"]=digest(STEP7/"config/l2x_semantic_amendment.json")
    else: payload["inner_groups"]="diag_model_id (included in frozen master table)"
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()


def semantic_amendment():
    path=STEP7/"config/l2x_semantic_amendment.json"
    if not path.exists(): return {"amendment":"not applied"}
    manifest=load(path)
    need(manifest["frozen_before_corrected_training"] and not manifest["candidate_selection_feedback"],"invalid semantic correction policy")
    need(digest(STEP7/"config/protocol_freeze.json")==manifest["original_protocol_sha256"],"original protocol changed")
    need(digest(ROOT/manifest["effective_table_path"])==manifest["effective_table_sha256"],"amended table changed")
    need(digest(STEP7/"scripts/amend_l2x.py")==manifest["code_sha256"],"amendment code changed")
    base=pd.read_parquet(STEP7/"analysis/phase5_master_table.parquet")
    corrected=pd.read_parquet(ROOT/manifest["effective_table_path"])
    changed=manifest["changed_columns"]
    mask=base.diag_parent_lineage.eq("Qwen")
    need(int(mask.sum())==276 and len(manifest["evidence"])==12,"amendment scope differs")
    pd.testing.assert_frame_equal(base.drop(columns=changed),corrected.drop(columns=changed))
    pd.testing.assert_frame_equal(base.loc[~mask,changed],corrected.loc[~mask,changed])
    need(corrected.loc[mask,changed].eq(0).all().all(),"disabled windows still imputed/unknown")
    frozen_static=pd.read_parquet(ROOT/"step3/analysis/model_architecture_features.parquet").set_index("model_id")
    for item in manifest["evidence"]:
        raw=ROOT/item["config_path"]
        need(hashlib.sha256(raw.read_bytes()).hexdigest()==item["raw_sha256"]==frozen_static.loc[item["model_id"],"config_sha256"],"window config integrity mismatch")
        config=load(raw)
        need(config.get("use_sliding_window") is False and config.get("sliding_window") is None,"window not explicitly disabled")
    lock=load(STEP7/"config/development_candidates_lock.json")
    need(digest(ROOT/manifest["original_matrix_path"])==manifest["original_matrix_sha256"]==lock["source_matrix_sha256"],"original candidate-selection evidence changed")
    old=pd.read_csv(ROOT/manifest["original_matrix_path"]).set_index("experiment_id")
    new=pd.read_csv(STEP7/"reports/main_model_matrix.csv").set_index("experiment_id")
    unaffected=old.feature_level.ne("L2X-StaticExtended")
    pd.testing.assert_frame_equal(old.loc[unaffected],new.loc[unaffected],check_exact=False,rtol=1e-12,atol=1e-10)
    return {"changed_rows":276,"models":12,"corrected_cells":16,"preserved_cells":208}


def reconcile_slice(rows,reported):
    need(len(rows)==int(reported["n_rows"]),"slice row count differs")
    need(rows.diag_model_id.nunique()==int(reported["n_models"]),"slice model count differs")
    need(np.isclose(np.median(rows.prediction-rows[TARGET]),reported["residual_median"],atol=1e-8),"slice median residual differs")
    for key,value in independent_metrics(rows[TARGET],rows.prediction,rows.x_hardware_num_gpus).items():
        need(np.isclose(value,reported[key],atol=1e-8,equal_nan=True),"slice metric differs: "+key)


def slice_statistics():
    """Rebuild all memberships and metrics independently of analyze_results.slices."""
    summaries=pd.read_csv(STEP7/"reports/slice_metrics.csv")
    need(not summaries.duplicated(["experiment_id","slice_type","slice_value"]).any(),"duplicate slice")
    checked=0
    for experiment,reported in summaries.groupby("experiment_id"):
        rows=pd.read_parquet(STEP7/"analysis"/f"oof_predictions_{experiment}.parquet")
        members={("overall","all"):rows}
        for kind,column in [("gpu_count","x_hardware_num_gpus"),("gpu_model","x_hardware_gpu_model"),("architecture","diag_architecture_class"),("parent_lineage","diag_parent_lineage"),("subfamily","diag_subfamily"),("task","x_protocol_task")]:
            for value,group in rows.groupby(column):
                members[(kind,str(value))]=group
        for kind,q in [("top25",.75),("top10",.9)]:
            cutoff=rows[TARGET].quantile(q)
            members[(kind,f">={cutoff:.6f}W")]=rows.loc[rows[TARGET]>=cutoff]
        bins=pd.qcut(rows[TARGET],4,labels=False,duplicates="drop")
        for value in sorted(bins.unique()):
            members[("power_quartile",f"Q{int(value)+1}")]=rows.loc[bins.eq(value)]
        need(set(zip(reported.slice_type,reported.slice_value.astype(str)))==set(members),"missing or mislabeled slices")
        for _,summary in reported.iterrows():
            reconcile_slice(members[(summary.slice_type,str(summary.slice_value))],summary)
            checked+=1
    need(summaries.experiment_id.nunique()==len(load(STEP7/"config/experiment_plan.json")["experiments"]),"slice experiments incomplete")
    return {"slice_cells_independently_rebuilt":checked}


def sources():
    freeze=load(STEP7/"config/source_freeze_phase5.json")
    for item in freeze["inputs"]:
        need(digest(ROOT/item["path"])==item["sha256"],f"source changed: {item['path']}")
    import os
    taskbook=Path(os.environ.get("MLENERGY_PHASE5_TASKBOOK",r"D:\my_work\cpecc\files\日常事务\科技项目\2025年\算电协同\课题一\阶段性研究进展\plan\first_step\Codex_Phase5_Taskbook.docx"))
    need(digest(taskbook)==freeze["taskbook_sha256"],"taskbook hash mismatch")
    protocol=load(STEP7/"config/protocol_freeze.json")
    need(protocol["frozen_before_training"],"protocol not frozen")
    for item in protocol["artifacts"]:
        need(digest(ROOT/item["path"])==item["sha256"],f"protocol changed: {item['path']}")
    return {"source_files":len(freeze["inputs"]),"protocol_files":len(protocol["artifacts"])}


def features():
    table=pd.read_parquet(STEP7/"analysis/phase5_master_table.parquet")
    original=pd.read_parquet(ROOT/"step6/analysis/phase4_power_modeling_table.parquet")
    need(len(table)==len(original)==565 and table.run_key.nunique()==565,"cohort count/grain differs")
    need(table.diag_model_id.nunique()==27,"model count differs")
    aligned=table[["run_key",TARGET]].merge(original[["run_key",TARGET]],on="run_key",validate="one_to_one",suffixes=("","_old"))
    need(np.array_equal(aligned[TARGET],aligned[TARGET+"_old"]),"frozen target changed")
    need(table.diag_quality_is_stable.all(),"unstable run in cohort")
    levels=load(STEP7/"config/feature_hierarchy_phase5.json")["levels"]
    patterns=("throughput","latency","duration","utilization","clock","temperature","measured_","actual_")
    for name,bundle in levels.items():
        columns=bundle["numeric"]+bundle["categorical"]
        for column in columns:
            need(column.startswith("x_") or column=="diag_architecture_class",f"identity/target X: {column}")
            need(not any(p in column for p in patterns),f"post-run X: {column}")
            need(column!="x_protocol_task" or name=="L1-TaskSensitivity","task label entered main X")
        if name in {"L0-Basic","L0A-ActiveAware","L1-Deployment"}:
            need(int(table[bundle["eligibility"]].sum())==565,"base level lost stable runs")
    need(np.allclose(table.x_p5_weight_bytes_total,table.x_model_benchmark_total_params_billions*1e9*table.x_p5_precision_bits/8),"weight bytes formula differs")
    model_parallel=table.x_hardware_num_gpus/table.x_deployment_data_parallel
    need((model_parallel>0).all() and (model_parallel%1==0).all(),"DP/num GPUs semantics invalid")
    need(np.allclose(table.x_p5_model_parallel_gpus_per_replica,model_parallel),"model-parallel group formula differs")
    need(np.allclose(table.x_p5_weight_hbm_ratio,table.x_p5_weight_bytes_total/model_parallel/(table.x_hardware_hbm_capacity_gb*1e9)),"weight HBM ratio differs")
    physics=table.loc[table.eligible_p5_l4]
    need(physics.x_model_attention_variant.eq("gqa").all(),"unsupported physics admitted")
    l,h,a,k,d=map(lambda c:physics[c].to_numpy(float),["x_model_num_layers","x_model_hidden_size","x_model_num_attention_heads","x_model_num_key_value_heads","x_model_head_dim"])
    mu=physics.x_workload_benchmark_input_tokens_mean.to_numpy(float)
    projection=4*l*h*d*(a+k)*mu
    attention=4*l*a*d*physics.x_workload_benchmark_input_tokens_second_moment
    kv=2*l*k*d*(physics.x_model_runtime_dtype_bits/8)
    need(np.allclose(projection,physics.x_p5_prefill_projection_flop_per_request),"projection formula differs")
    need(np.allclose(attention,physics.x_p5_prefill_attention_flop_per_request),"attention formula differs")
    need(np.allclose(kv*mu,physics.x_p5_kv_mean_request_bytes),"KV formula differs")
    return {"stable_rows":len(table),"models":27,"levels":len(levels),"independent_physics_rows":len(physics)}


def folds_and_candidates():
    plan=load(STEP7/"config/experiment_plan.json")
    sampled=load(STEP7/"config/sampled_hyperparameter_candidates.json")["algorithms"]
    need(len(sampled)==8,"algorithm count differs")
    for algorithm,candidates in sampled.items():
        need(len(candidates)==(1 if algorithm=="DummyMean" else 8),"unequal candidate budget")
    for cohort in plan["cohorts"]:
        manifest=pd.read_csv(STEP7/"splits"/f"outer_model_folds_{cohort}.csv")
        need(manifest.run_key.nunique()==len(manifest),"duplicate split row")
        need(manifest.outer_fold.nunique()==5,"outer count differs")
        need(manifest.groupby("diag_model_id").outer_fold.nunique().eq(1).all(),"model ID crosses outer folds")
    need({s["target_strategy"] for s in plan["experiments"]}=={"T-total","G-perGPU"},"target strategy differs")
    return {"cohorts":len(plan["cohorts"]),"experiments":len(plan["experiments"]),"nonbaseline_budget":8}


def oof_predictions():
    plan=load(STEP7/"config/experiment_plan.json")
    matrix=pd.read_csv(STEP7/"reports/main_model_matrix.csv")
    need(len(matrix)==len(plan["experiments"]),"matrix incomplete")
    rebuilt=0
    for spec in plan["experiments"]:
        rows=pd.read_parquet(STEP7/"analysis"/f"oof_predictions_{spec['experiment_id']}.parquet")
        need("checkpoint_fingerprint" in rows and set(rows.checkpoint_fingerprint)=={independent_fingerprint(spec)},"OOF fingerprint differs from current original/amended provenance")
        folds=pd.read_csv(STEP7/"splits"/f"outer_model_folds_{spec['cohort']}.csv")
        need(len(rows)==len(folds) and set(rows.run_key)==set(folds.run_key),"OOF support differs")
        pairs=rows[["run_key","outer_fold","diag_model_id"]].merge(folds[["run_key","outer_fold","diag_model_id"]],on=["run_key","outer_fold","diag_model_id"],validate="one_to_one")
        need(len(pairs)==len(rows),"OOF fold/key mismatch")
        need(np.isfinite(rows.prediction).all(),"nonfinite predictions")
        need(np.allclose(rows.prediction_per_gpu*rows.x_hardware_num_gpus,rows.prediction),"per-GPU restoration mismatch")
        summary=matrix.loc[matrix.experiment_id.eq(spec["experiment_id"])]
        need(len(summary)==1,"duplicate matrix identity")
        for key,value in independent_metrics(rows[TARGET],rows.prediction,rows.x_hardware_num_gpus).items():
            need(np.isclose(value,summary.iloc[0][key],rtol=1e-10,atol=1e-8,equal_nan=True),f"metric differs: {spec['experiment_id']} {key}")
        for fold,group in rows.groupby("outer_fold"):
            metadata=json.loads(group.selection_metadata_json.iloc[0])
            sampled=load(STEP7/"config/sampled_hyperparameter_candidates.json")["algorithms"][spec["algorithm"]]
            need(len(metadata["inner_scores"])==len(sampled),"candidate count differs from frozen search")
            need(metadata["selected_hyperparameters"]==sampled[metadata["selected_index"]],"selected hyperparameters not frozen candidate")
            need(metadata["algorithm"]==spec["algorithm"] and metadata["target_strategy"]==spec["target_strategy"],"wrong algorithm/target metadata")
            need(set(metadata["train_models"])==set(folds.loc[folds.outer_fold.ne(fold),"diag_model_id"]),"outer train support differs")
            need(not set(metadata["train_models"])&set(metadata["test_models"]),"train/test model overlap")
            need(set(metadata["test_models"])==set(group.diag_model_id),"metadata test scope differs")
            need(metadata["selection_metric"]=="row_weighted_reconstructed_total_mae","wrong target-selection scale")
            score=[float(x) if x is not None else np.inf for x in metadata["inner_scores"]]
            need(metadata["selected_index"]==int(np.argmin(score)),"wrong inner-CV candidate")
            need(len(metadata["inner_groups"])==3,"inner fold count differs")
            validations=[]
            for assignment in metadata["inner_groups"]:
                need(not set(assignment["train"])&set(assignment["validation"]),"inner model group overlap")
                need(not set(metadata["test_models"])&(set(assignment["train"])|set(assignment["validation"])),"outer test entered inner CV")
                need(set(assignment["train"])|set(assignment["validation"])==set(metadata["train_models"]),"inner scope lost models")
                validations.extend(assignment["validation"])
            need(len(validations)==len(set(validations)) and set(validations)==set(metadata["train_models"]),"inner held-out coverage differs")
        rebuilt+=1
    return {"cells_independently_rebuilt":rebuilt}


def paired_statistics():
    effects=pd.read_csv(STEP7/"reports/feature_increment_effects.csv")
    targets=pd.read_csv(STEP7/"reports/target_representation_comparison.csv")
    draw_tables={"feature":pd.read_parquet(STEP7/"analysis/paired_bootstrap_features.parquet"),"target":pd.read_parquet(STEP7/"analysis/paired_bootstrap_targets.parquet")}
    rebuilt=0
    def oof(level,cohort,algorithm,strategy):
        return pd.read_parquet(STEP7/"analysis"/f"oof_predictions_{level}__{cohort}__{algorithm}__{strategy}.parquet")
    for kind,table in [("feature",effects),("target",targets)]:
        for _,reported in table.iterrows():
            if kind=="feature":
                before=oof(reported.lower,reported.cohort,reported.algorithm,reported.target_strategy)
                after=oof(reported.higher,reported.cohort,reported.algorithm,reported.target_strategy)
                selected=draw_tables[kind].loc[draw_tables[kind].comparison.eq(reported.comparison)&draw_tables[kind].algorithm.eq(reported.algorithm)&draw_tables[kind].target_strategy.eq(reported.target_strategy)]
            else:
                before=oof(reported.feature_level,reported.cohort,reported.algorithm,"T-total")
                after=oof(reported.feature_level,reported.cohort,reported.algorithm,"G-perGPU")
                selected=draw_tables[kind].loc[draw_tables[kind].feature_level.eq(reported.feature_level)&draw_tables[kind].cohort.eq(reported.cohort)&draw_tables[kind].algorithm.eq(reported.algorithm)]
            identity=["run_key","diag_model_id","outer_fold",TARGET]
            pairs=before[identity+["prediction"]].merge(after[identity+["prediction"]],on=identity,suffixes=("_a","_b"),validate="one_to_one")
            need(len(pairs)==len(before)==len(after),"paired support/folds differ")
            delta=(pairs.prediction_b-pairs[TARGET]).abs()-(pairs.prediction_a-pairs[TARGET]).abs()
            need(np.isclose(delta.mean(),reported.row_weighted_delta_mae,atol=1e-8),"paired row delta differs")
            grouped=delta.groupby(pairs.diag_model_id).agg(["sum","count","mean"]).sort_index()
            need(np.isclose(grouped["mean"].mean(),reported.equal_model_delta_mae,atol=1e-8),"paired equal-model delta differs")
            need(len(selected)==int(reported.bootstrap_replicates)>=1000,"bootstrap support differs")
            for column,lo,hi in [("row_weighted_delta_mae","row_ci_low","row_ci_high"),("equal_model_delta_mae","equal_model_ci_low","equal_model_ci_high")]:
                need(np.isclose(np.quantile(selected[column],.025),reported[lo],atol=1e-8),"CI low differs")
                need(np.isclose(np.quantile(selected[column],.975),reported[hi],atol=1e-8),"CI high differs")
            rng=np.random.default_rng(20261013)
            selections=rng.integers(0,len(grouped),size=(len(selected),len(grouped)))
            row_draws=grouped["sum"].to_numpy()[selections].sum(axis=1)/grouped["count"].to_numpy()[selections].sum(axis=1)
            need(np.allclose(row_draws,selected.sort_values("draw").row_weighted_delta_mae,atol=1e-8),"draws are not model-cluster bootstrap")
            rebuilt+=1
    return {"paired_comparisons_rebuilt":rebuilt,"draws":sum(len(x) for x in draw_tables.values())}


def candidate_and_stress():
    lock_path=STEP7/"config/development_candidates_lock.json"
    lock=load(lock_path)
    need(lock["development_candidate_not_external_validated"],"false external validation claim")
    for level,choice in lock["choices"].items():
        metadata=load(STEP7/"models/development_candidates"/f"{level}.json")
        model_path=STEP7/"models/development_candidates"/f"{level}.joblib"
        need(digest(model_path)==metadata["model_sha256"],"candidate model changed")
        need(digest(lock_path)==metadata["lock_sha256"],"candidate lock changed")
        need(metadata["development_candidate_not_external_validated"],"candidate metadata validation claim")
    summary=pd.read_csv(STEP7/"reports/generalization_stress.csv")
    predictions=pd.read_parquet(STEP7/"analysis/post_hoc_stress_predictions.parquet")
    need(summary.post_hoc_only.all() and predictions.post_hoc_only.all(),"stress lost post-hoc label")
    for (level,scheme,fold),rows in predictions.groupby(["feature_level","stress_scheme","stress_fold"]):
        manifest=pd.read_csv(STEP7/"splits"/f"post_hoc_{level}_{scheme}.csv")
        candidate_metadata=load(STEP7/"models/development_candidates"/f"{level}.json")
        payload={"candidate_lock":digest(lock_path),"metadata":candidate_metadata,"manifest":digest(STEP7/"splits"/f"post_hoc_{level}_{scheme}.csv"),"code":digest(STEP7/"scripts/run_stress.py"),"versions":candidate_metadata["versions"],"dependencies":{"modeling.py":digest(STEP7/"scripts/modeling.py"),"common.py":digest(STEP7/"scripts/common.py"),"protocol":digest(STEP7/"config/protocol_freeze.json"),"source_freeze":digest(STEP7/"config/source_freeze_phase5.json")}}
        expected=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
        need("checkpoint_fingerprint" in rows and set(rows.checkpoint_fingerprint)=={expected},"stress fingerprint mismatch (shared code/protocol)")
        assigned=manifest.loc[manifest.stress_fold.eq(fold)]
        need(set(rows.run_key)==set(assigned.run_key) and len(rows)==len(assigned),"stress manifest mismatch")
        reported=summary.loc[summary.feature_level.eq(level)&summary.scheme.eq(scheme)&summary.held_out_group.eq(rows.held_out_group.iloc[0])]
        if scheme=="within_task_lineage":
            reported=reported.loc[reported.task.eq(rows.x_protocol_task.iloc[0])]
        need(len(reported)==1,"stress summary ambiguous")
        for key,value in independent_metrics(rows[TARGET],rows.prediction,rows.x_hardware_num_gpus).items():
            need(np.isclose(value,reported.iloc[0][key],atol=1e-8,equal_nan=True),"stress metric mismatch")
    return {"development_candidates":len(lock["choices"]),"post_hoc_stress_cells":len(summary)}


def artifacts_and_secrets():
    required=["README.md","reports/prior_evidence_summary.md","reports/phase5_hierarchical_modeling_report.md","reports/feature_semantics_audit.md","reports/physics_feature_formulas_phase5.md","analysis/feature_dictionary_phase5.csv","reports/feature_dictionary_phase5.md","reports/figure_sources.json","reports/slice_metrics.csv","notebooks/phase5_hierarchical_modeling.ipynb","notebooks/phase5_hierarchical_modeling.html"]
    for p in required:
        need((STEP7/p).exists(),"missing required artifact: "+p)
    dictionary=pd.read_csv(STEP7/"analysis/feature_dictionary_phase5.csv")
    hierarchy=load(STEP7/"config/feature_hierarchy_phase5.json")["levels"]
    expected={(c,n) for n,b in hierarchy.items() for c in b["columns"]}
    need(set(zip(dictionary.feature,dictionary.feature_level))==expected and len(dictionary)==len(expected),"feature dictionary incomplete")
    need(dictionary[["source","unit","formula","applicability","exante_reason","limitation"]].notna().all().all(),"undefined feature semantics")
    figures=["information_accuracy_curve.png","algorithm_feature_heatmap.png","total_per_gpu_paired.png","gpu_count_error_scaling.png","calibration_overall_8gpu_top10.png","feature_increment_forest.png"]
    for name in figures:
        need((STEP7/"figures"/name).stat().st_size>5000,"trivial figure: "+name)
    notebook=load(STEP7/"notebooks/phase5_hierarchical_modeling.ipynb")
    cells=[c for c in notebook["cells"] if c["cell_type"]=="code"]
    need(cells and all(c.get("execution_count") is not None for c in cells),"unexecuted notebook")
    need(not any(o.get("output_type")=="error" for c in cells for o in c.get("outputs",[])),"notebook error")
    source="\n".join("".join(c["source"]) for c in cells)
    need(".fit(" not in source and "joblib.load" not in source,"notebook retrains/loads model")
    patterns=[re.compile(r"hf"+r"_[A-Za-z0-9]{20,}"),re.compile(r"gh"+r"[pousr]_[A-Za-z0-9]{30,}"),re.compile(r"github"+r"_pat_[A-Za-z0-9_]{20,}"),re.compile(r"sk"+r"-[A-Za-z0-9_-]{20,}")]
    scanned=0
    for path in STEP7.rglob("*"):
        if not path.is_file() or "checkpoints" in path.parts or path.suffix.lower() not in {".py",".json",".md",".csv",".ipynb",".html",".txt"}:
            continue
        content=path.read_text(encoding="utf-8",errors="ignore")
        need(not any(p.search(content) for p in patterns),f"credential-like content: {path.relative_to(STEP7)}")
        scanned+=1
    return {"required_artifacts":len(required),"figures":len(figures),"executed_code_cells":len(cells),"secret_pattern_files":scanned}


def run_validations(*,write=True,progress_only=False):
    checks=[]
    pairs=[("source/protocol integrity",sources),("cohort/feature/formula boundary",features),("fold/candidate fairness",folds_and_candidates)]
    if (STEP7/"config/l2x_semantic_amendment.json").exists():
        pairs.append(("selective semantic amendment and preserved main evidence",semantic_amendment))
    if not progress_only:
        pairs += [("OOF total/per-GPU metrics and isolation",oof_predictions),("slice memberships and both power scales",slice_statistics),("paired model-cluster statistics",paired_statistics),("candidate freeze and stress evidence",candidate_and_stress),("required delivery and credential patterns",artifacts_and_secrets)]
    for name,function in pairs:
        try:
            checks.append({"name":name,"status":"pass","details":function()})
        except Exception as exc:
            checks.append({"name":name,"status":"fail","error":f"{type(exc).__name__}: {exc}"})
    receipt={"schema_version":"1.0","scope":"Phase5 GPU-side stable high-load text-LLM inference","mode":"progress_only" if progress_only else "final","status":"pass" if all(c["status"]=="pass" for c in checks) else "fail","generated_utc":datetime.now(timezone.utc).isoformat(),"checks":checks}
    if write:
        output=STEP7/"reports"/("progress_validation.json" if progress_only else "verification_receipt.json")
        output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--progress-only",action="store_true")
    parser.add_argument("--no-write",action="store_true")
    args=parser.parse_args()
    result=run_validations(write=not args.no_write,progress_only=args.progress_only)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result["status"]!="pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
