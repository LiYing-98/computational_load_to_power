"""Matched-support increments, target effects and correctly scaled error slices."""
import argparse
import json

import numpy as np
import pandas as pd

from step7.scripts.common import STEP7, TARGET, SEED, read_json, write_json
from step7.scripts.modeling import metrics
from step7.scripts.freeze_protocol import ALGORITHMS, verify_protocol


def paired_effect(before, after, *, n_bootstrap=2000):
    identity = ["run_key","diag_model_id","outer_fold",TARGET]
    if set(before.run_key)!=set(after.run_key) or before.run_key.duplicated().any() or after.run_key.duplicated().any():
        raise ValueError("paired comparison requires identical unique run support")
    paired = before[identity+["prediction"]].merge(after[identity+["prediction"]],on=identity,suffixes=("_before","_after"),validate="one_to_one")
    if len(paired)!=len(before):
        raise ValueError("paired folds/model identities/targets do not match")
    paired["error_delta"] = (paired.prediction_after-paired[TARGET]).abs()-(paired.prediction_before-paired[TARGET]).abs()
    clusters = paired.groupby("diag_model_id").error_delta.agg(["sum","count","mean"]).sort_index()
    rng = np.random.default_rng(SEED)
    chosen = rng.integers(0,len(clusters),size=(n_bootstrap,len(clusters)))
    row_delta = clusters["sum"].to_numpy()[chosen].sum(axis=1)/clusters["count"].to_numpy()[chosen].sum(axis=1)
    equal_delta = clusters["mean"].to_numpy()[chosen].mean(axis=1)
    summary = {"n_rows":len(paired),"n_models":len(clusters),"row_weighted_delta_mae":float(paired.error_delta.mean()),"equal_model_delta_mae":float(clusters["mean"].mean()),"row_ci_low":float(np.quantile(row_delta,.025)),"row_ci_high":float(np.quantile(row_delta,.975)),"equal_model_ci_low":float(np.quantile(equal_delta,.025)),"equal_model_ci_high":float(np.quantile(equal_delta,.975)),"bootstrap_replicates":n_bootstrap}
    draws = pd.DataFrame({"draw":np.arange(n_bootstrap),"row_weighted_delta_mae":row_delta,"equal_model_delta_mae":equal_delta})
    return summary,draws


def prediction_path(level,cohort,algorithm,strategy):
    return STEP7/"analysis"/f"oof_predictions_{level}__{cohort}__{algorithm}__{strategy}.parquet"


def slices(oof):
    output = []
    definitions = [("overall","all",oof)]
    for kind,column in [("gpu_count","x_hardware_num_gpus"),("gpu_model","x_hardware_gpu_model"),("architecture","diag_architecture_class"),("parent_lineage","diag_parent_lineage"),("subfamily","diag_subfamily"),("task","x_protocol_task")]:
        for value, group in oof.groupby(column):
            definitions.append((kind,str(value),group))
    for kind,quantile in [("top25",.75),("top10",.9)]:
        threshold = oof[TARGET].quantile(quantile)
        definitions.append((kind,f">={threshold:.6f}W",oof.loc[oof[TARGET]>=threshold]))
    quartiles = pd.qcut(oof[TARGET],4,labels=False,duplicates="drop")
    for bin_id in sorted(quartiles.unique()):
        definitions.append(("power_quartile",f"Q{int(bin_id)+1}",oof.loc[quartiles.eq(bin_id)]))
    spec_columns = ["experiment_id","feature_level","algorithm","target_strategy","cohort","analysis_role"]
    labels = {c:oof[c].iloc[0] for c in spec_columns}
    for kind,value,group in definitions:
        output.append({**labels,"slice_type":kind,"slice_value":value,"n_rows":len(group),"n_models":group.diag_model_id.nunique(),"residual_median":float(group.residual.median()),**metrics(group[TARGET],group.prediction,group.x_hardware_num_gpus)})
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--allow-partial",action="store_true")
    args=parser.parse_args()
    verify_protocol()
    plan=read_json(STEP7/"config/experiment_plan.json")
    missing=[s["experiment_id"] for s in plan["experiments"] if not prediction_path(s["feature_level"],s["cohort"],s["algorithm"],s["target_strategy"]).exists()]
    if missing and not args.allow_partial:
        raise ValueError(f"matrix incomplete: {len(missing)}/{len(plan['experiments'])}; no final statistics emitted")
    feature_rows,target_rows,feature_draws,target_draws,slice_rows=[],[],[],[],[]
    for pair in plan["adjacent_pairs"]:
        for algorithm in ALGORITHMS:
            for strategy in ["T-total","G-perGPU"]:
                before_path=prediction_path(pair["lower"],pair["cohort"],algorithm,strategy)
                after_path=prediction_path(pair["higher"],pair["cohort"],algorithm,strategy)
                if not before_path.exists() or not after_path.exists():
                    continue
                summary,draws=paired_effect(pd.read_parquet(before_path),pd.read_parquet(after_path))
                labels={**pair,"algorithm":algorithm,"target_strategy":strategy,"comparison":pair["lower"]+"_to_"+pair["higher"]}
                feature_rows.append({**labels,**summary})
                feature_draws.append(draws.assign(**labels))
    levels_support=sorted({(s["feature_level"],s["cohort"]) for s in plan["experiments"]})
    for level,cohort in levels_support:
        for algorithm in ALGORITHMS:
            before_path=prediction_path(level,cohort,algorithm,"T-total")
            after_path=prediction_path(level,cohort,algorithm,"G-perGPU")
            if not before_path.exists() or not after_path.exists():
                continue
            summary,draws=paired_effect(pd.read_parquet(before_path),pd.read_parquet(after_path))
            labels={"feature_level":level,"cohort":cohort,"algorithm":algorithm,"comparison":"G-perGPU_minus_T-total"}
            target_rows.append({**labels,**summary})
            target_draws.append(draws.assign(**labels))
    for spec in plan["experiments"]:
        path=prediction_path(spec["feature_level"],spec["cohort"],spec["algorithm"],spec["target_strategy"])
        if path.exists():
            slice_rows.extend(slices(pd.read_parquet(path)))
    for filename,rows in [("feature_increment_effects.csv",feature_rows),("target_representation_comparison.csv",target_rows),("slice_metrics.csv",slice_rows)]:
        pd.DataFrame(rows).to_csv(STEP7/"reports"/filename,index=False)
    if feature_draws:
        pd.concat(feature_draws,ignore_index=True).to_parquet(STEP7/"analysis/paired_bootstrap_features.parquet",index=False)
    if target_draws:
        pd.concat(target_draws,ignore_index=True).to_parquet(STEP7/"analysis/paired_bootstrap_targets.parquet",index=False)
    write_json(STEP7/"analysis/statistical_analysis_status.json",{"status":"partial" if missing else "complete","feature_pairs":len(feature_rows),"target_pairs":len(target_rows),"slice_rows":len(slice_rows),"missing_experiments":len(missing)})
    print(json.dumps({"feature_pairs":len(feature_rows),"target_pairs":len(target_rows),"slice_rows":len(slice_rows),"missing":len(missing)}))


if __name__ == "__main__":
    main()
