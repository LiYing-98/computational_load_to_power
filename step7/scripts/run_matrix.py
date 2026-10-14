"""Resume input-fingerprinted Phase5 OOF cells; no partial-cell success metrics."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from step7.scripts.common import ROOT, STEP7, TARGET, read_json, sha256, write_json
from step7.scripts.freeze_protocol import verify_protocol
from step7.scripts.modeling import fit_outer, metrics, versions


def checkpoint_key(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()


def valid_checkpoint(path, fingerprint, test_keys):
    if not Path(path).exists():
        return False
    frame = pd.read_parquet(path)
    return "checkpoint_fingerprint" in frame and set(frame.checkpoint_fingerprint)=={fingerprint} and frame.run_key.nunique()==len(test_keys) and len(frame)==len(test_keys) and set(frame.run_key)==test_keys


def experiment_fingerprint(spec):
    protocol = verify_protocol()
    return checkpoint_key({
        "spec":spec,"protocol":protocol,"source_freeze":sha256(STEP7/"config/source_freeze_phase5.json"),
        "versions":versions(),"code":{name:sha256(STEP7/"scripts"/name) for name in ["run_matrix.py","modeling.py","common.py","freeze_protocol.py"]},
        "row_order":"run_key ascending; fit and hash both canonical",
        "inner_groups":"diag_model_id (included in frozen master table)",
    })


def run_experiment(spec):
    started = time.monotonic()
    fingerprint = experiment_fingerprint(spec)
    output_path = STEP7/"analysis"/f"oof_predictions_{spec['experiment_id']}.parquet"
    frame = pd.read_parquet(STEP7/"analysis/phase5_master_table.parquet").sort_values("run_key").reset_index(drop=True)
    folds = pd.read_csv(STEP7/"splits"/f"outer_model_folds_{spec['cohort']}.csv")
    support_keys = set(folds.run_key)
    frame = frame.loc[frame.run_key.isin(support_keys)].reset_index(drop=True)
    if valid_checkpoint(output_path,fingerprint,support_keys):
        return {"experiment_id":spec["experiment_id"],"status":"reused", "rows":len(frame)}
    bundle = read_json(STEP7/"config/feature_hierarchy_phase5.json")["levels"][spec["feature_level"]]
    if not frame[bundle["eligibility"]].all():
        raise ValueError(f"level not eligible on matched support: {spec}")
    candidates = read_json(STEP7/"config/sampled_hyperparameter_candidates.json")["algorithms"][spec["algorithm"]]
    checkpoint_dir = STEP7/"models/checkpoints"/fingerprint
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    chunks = []
    for fold in sorted(folds.outer_fold.unique()):
        test_keys = set(folds.loc[folds.outer_fold.eq(fold),"run_key"])
        checkpoint = checkpoint_dir/f"fold_{fold}.parquet"
        if valid_checkpoint(checkpoint,fingerprint,test_keys):
            chunks.append(pd.read_parquet(checkpoint))
            continue
        test = frame.loc[frame.run_key.isin(test_keys)].copy()
        train = frame.loc[~frame.run_key.isin(test_keys)].copy()
        pred,info,_ = fit_outer(train,test,spec["algorithm"],bundle,candidates,spec["target_strategy"])
        columns = ["run_key","diag_model_id","diag_parent_lineage","diag_subfamily","diag_architecture_class","x_protocol_task","x_hardware_num_gpus","x_hardware_gpu_model",TARGET]
        result = test[columns].copy()
        result["prediction"] = pred
        result["prediction_per_gpu"] = pred/test.x_hardware_num_gpus.to_numpy(float)
        result["residual"] = pred-test[TARGET].to_numpy(float)
        result["outer_fold"] = int(fold)
        for key,value in spec.items():
            result[key] = value
        result["checkpoint_fingerprint"] = fingerprint
        result["selection_metadata_json"] = json.dumps(info,sort_keys=True,separators=(",",":"))
        result.to_parquet(checkpoint,index=False)
        chunks.append(result)
        print(f"fold {fold+1}/5 complete: {spec['experiment_id']}",flush=True)
    oof = pd.concat(chunks).sort_values("run_key").reset_index(drop=True)
    if len(oof)!=len(frame) or oof.run_key.nunique()!=len(frame):
        raise ValueError("incomplete/duplicate OOF; refusing aggregate metrics")
    oof.to_parquet(output_path,index=False)
    return {"experiment_id":spec["experiment_id"],"status":"completed","rows":len(oof),"seconds":round(time.monotonic()-started,2)}


def collect_results():
    plan = read_json(STEP7/"config/experiment_plan.json")
    rows,missing = [],[]
    for spec in plan["experiments"]:
        path = STEP7/"analysis"/f"oof_predictions_{spec['experiment_id']}.parquet"
        if not path.exists():
            missing.append(spec["experiment_id"])
            continue
        oof = pd.read_parquet(path)
        current = experiment_fingerprint(spec)
        if set(oof.checkpoint_fingerprint)!={current}:
            missing.append(spec["experiment_id"])
            continue
        folds = pd.read_csv(STEP7/"splits"/f"outer_model_folds_{spec['cohort']}.csv")
        if len(oof)!=len(folds) or set(oof.run_key)!=set(folds.run_key):
            raise ValueError(f"partial cell: {spec['experiment_id']}")
        rows.append({**spec,"n_rows":len(oof),"n_models":oof.diag_model_id.nunique(),**metrics(oof[TARGET],oof.prediction,oof.x_hardware_num_gpus)})
    if rows:
        pd.DataFrame(rows).to_csv(STEP7/"reports/main_model_matrix.csv",index=False)
    write_json(STEP7/"analysis/matrix_progress.json",{"status":"complete" if not missing else "in_progress","completed_experiments":len(rows),"expected_experiments":len(plan["experiments"]),"missing_experiments":missing})
    return len(rows),len(plan["experiments"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--algorithms",nargs="*")
    parser.add_argument("--collect-only",action="store_true")
    args = parser.parse_args()
    verify_protocol()
    plan = read_json(STEP7/"config/experiment_plan.json")
    if args.collect_only:
        print(collect_results())
        return
    specs = [s for s in plan["experiments"] if not args.algorithms or s["algorithm"] in args.algorithms]
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_experiment,spec):spec for spec in specs}
        for future in as_completed(pending):
            spec = pending[future]
            try:
                result = future.result()
                print(json.dumps(result),flush=True)
            except Exception as exc:
                item={"experiment_id":spec["experiment_id"],"status":"failed","error":f"{type(exc).__name__}: {exc}"}
                failures.append(item)
                print(json.dumps(item),flush=True)
            write_json(STEP7/"analysis/matrix_failures.json",failures)
    completed,total = collect_results()
    print(json.dumps({"completed_experiments":completed,"expected_experiments":total,"failures":failures},indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
