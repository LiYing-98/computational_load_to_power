"""Explicit selective semantic repair; original frozen protocol/master remain intact."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import shutil

import pandas as pd

from step7.scripts.common import ROOT,STEP7,TARGET,read_json,write_json,sha256
from step7.scripts.freeze_protocol import verify_protocol
from step7.scripts.modeling import fit_outer,metrics,versions
from step7.scripts.run_matrix import checkpoint_key,valid_checkpoint,experiment_fingerprint

AMENDMENT=STEP7/"config/l2x_semantic_amendment.json"
TABLE=STEP7/"analysis/phase5_l2x_amended_table.parquet"
ARCHIVE=STEP7/"analysis/superseded_l2x"
CHANGED=["x_p5_extended_sliding_window","x_p5_extended_sliding_window_applicable","x_p5_extended_sliding_window_unknown"]


def corrected_table(frame):
    output=frame.copy()
    static=pd.read_parquet(ROOT/"step3/analysis/model_architecture_features.parquet").set_index("model_id")
    evidence=[]
    for model in sorted(frame.loc[frame.diag_parent_lineage.eq("Qwen"),"diag_model_id"].unique()):
        path=ROOT/"step3/data/static_models"/model.replace("/","--")/"config.json"
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=static.loc[model,"config_sha256"]:
            raise ValueError("raw config differs from frozen architecture evidence: "+model)
        config=read_json(path)
        if config.get("use_sliding_window") is not False or config.get("sliding_window") is not None:
            raise ValueError("explicit disabled/null rule not satisfied: "+model)
        mask=frame.diag_model_id.eq(model)
        output.loc[mask,CHANGED]=0.
        evidence.append({"model_id":model,"rows":int(mask.sum()),"config_path":path.relative_to(ROOT).as_posix(),"raw_sha256":digest,"use_sliding_window":False,"sliding_window":None})
    return output,evidence


def freeze_amendment():
    verify_protocol()
    corrected,evidence=corrected_table(pd.read_parquet(STEP7/"analysis/phase5_master_table.parquet"))
    if AMENDMENT.exists():
        frozen=read_json(AMENDMENT)
        if frozen["code_sha256"]!=sha256(__file__) or frozen["effective_table_sha256"]!=sha256(TABLE) or frozen["evidence"]!=evidence:
            raise ValueError("frozen amendment changed")
        return frozen
    ARCHIVE.mkdir(exist_ok=True)
    shutil.copy2(STEP7/"reports/main_model_matrix.csv",ARCHIVE/"original_main_model_matrix.csv")
    for path in (STEP7/"analysis").glob("oof_predictions_L2X-StaticExtended__*.parquet"):
        shutil.copy2(path,ARCHIVE/path.name)
    corrected.to_parquet(TABLE,index=False)
    frozen={"schema_version":"1.0","frozen_before_corrected_training":True,"frozen_utc":datetime.now(timezone.utc).isoformat(),"reason":"review-confirmed structural N/A semantic bug; not result-based retuning","affected_level":"L2X-StaticExtended","changed_columns":CHANGED,"rule":"existing pinned raw config use_sliding_window=false and sliding_window=null -> value0, applicable0, unknown0","evidence":evidence,"original_protocol_sha256":sha256(STEP7/"config/protocol_freeze.json"),"original_master_sha256":sha256(STEP7/"analysis/phase5_master_table.parquet"),"effective_table_path":TABLE.relative_to(ROOT).as_posix(),"effective_table_sha256":sha256(TABLE),"code_sha256":sha256(__file__),"original_matrix_path":(ARCHIVE/"original_main_model_matrix.csv").relative_to(ROOT).as_posix(),"original_matrix_sha256":sha256(ARCHIVE/"original_main_model_matrix.csv"),"candidate_selection_feedback":False}
    write_json(AMENDMENT,frozen)
    return frozen


def amended_fingerprint(spec):
    frozen=read_json(AMENDMENT)
    if frozen["code_sha256"]!=sha256(__file__) or frozen["effective_table_sha256"]!=sha256(TABLE):
        raise ValueError("amendment artifact mismatch")
    return checkpoint_key({"spec":spec,"protocol":verify_protocol(),"source_freeze":sha256(STEP7/"config/source_freeze_phase5.json"),"amendment":sha256(AMENDMENT),"versions":versions(),"code":{name:sha256(STEP7/"scripts"/name) for name in ["amend_l2x.py","modeling.py","common.py","freeze_protocol.py"]},"row_order":"run_key ascending; same frozen support and nested groups"})


def run_cell(spec):
    fingerprint=amended_fingerprint(spec)
    folds=pd.read_csv(STEP7/"splits"/f"outer_model_folds_{spec['cohort']}.csv")
    keys=set(folds.run_key)
    destination=STEP7/"analysis"/f"oof_predictions_{spec['experiment_id']}.parquet"
    if valid_checkpoint(destination,fingerprint,keys):
        return spec["experiment_id"]+" reused"
    frame=pd.read_parquet(TABLE).sort_values("run_key").reset_index(drop=True)
    frame=frame.loc[frame.run_key.isin(keys)].copy()
    bundle=read_json(STEP7/"config/feature_hierarchy_phase5.json")["levels"][spec["feature_level"]]
    candidates=read_json(STEP7/"config/sampled_hyperparameter_candidates.json")["algorithms"][spec["algorithm"]]
    chunks=[]
    for fold in sorted(folds.outer_fold.unique()):
        test_keys=set(folds.loc[folds.outer_fold.eq(fold),"run_key"])
        path=STEP7/"models/checkpoints"/fingerprint/f"fold_{fold}.parquet"
        if valid_checkpoint(path,fingerprint,test_keys):
            chunks.append(pd.read_parquet(path)); continue
        test=frame.loc[frame.run_key.isin(test_keys)].copy()
        train=frame.loc[~frame.run_key.isin(test_keys)].copy()
        pred,info,_=fit_outer(train,test,spec["algorithm"],bundle,candidates,spec["target_strategy"])
        result=test[["run_key","diag_model_id","diag_parent_lineage","diag_subfamily","diag_architecture_class","x_protocol_task","x_hardware_num_gpus","x_hardware_gpu_model",TARGET]].copy()
        result["prediction"]=pred
        result["prediction_per_gpu"]=pred/test.x_hardware_num_gpus.to_numpy(float)
        result["residual"]=pred-test[TARGET].to_numpy(float)
        result["outer_fold"]=int(fold)
        for key,value in spec.items(): result[key]=value
        result["checkpoint_fingerprint"]=fingerprint
        result["selection_metadata_json"]=json.dumps(info,sort_keys=True,separators=(",",":"))
        result["semantic_amendment"]="l2x_disabled_sliding_window_v1"
        path.parent.mkdir(parents=True,exist_ok=True)
        result.to_parquet(path,index=False)
        chunks.append(result)
    pd.concat(chunks).sort_values("run_key").reset_index(drop=True).to_parquet(destination,index=False)
    return spec["experiment_id"]+" corrected"


def collect_amended_results():
    plan=read_json(STEP7/"config/experiment_plan.json")
    rows=[]
    for spec in plan["experiments"]:
        path=STEP7/"analysis"/f"oof_predictions_{spec['experiment_id']}.parquet"
        folds=pd.read_csv(STEP7/"splits"/f"outer_model_folds_{spec['cohort']}.csv")
        expected=amended_fingerprint(spec) if spec["feature_level"]=="L2X-StaticExtended" else experiment_fingerprint(spec)
        if not valid_checkpoint(path,expected,set(folds.run_key)):
            raise ValueError("incomplete/stale current cell: "+spec["experiment_id"])
        oof=pd.read_parquet(path)
        rows.append({**spec,"n_rows":len(oof),"n_models":oof.diag_model_id.nunique(),**metrics(oof[TARGET],oof.prediction,oof.x_hardware_num_gpus)})
    pd.DataFrame(rows).to_csv(STEP7/"reports/main_model_matrix.csv",index=False)
    write_json(STEP7/"analysis/matrix_progress.json",{"status":"complete","completed_experiments":len(rows),"expected_experiments":len(plan["experiments"]),"missing_experiments":[],"semantic_amendment":"config/l2x_semantic_amendment.json","collector":"step7.scripts.amend_l2x.collect_amended_results; original collector is pre-amendment only"})


def main():
    freeze_amendment()
    specs=[s for s in read_json(STEP7/"config/experiment_plan.json")["experiments"] if s["feature_level"]=="L2X-StaticExtended"]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(run_cell,s) for s in specs]):
            print(future.result(),flush=True)
    collect_amended_results()
    print("16 L2X cells amended; 208 original cells retained",flush=True)


if __name__=="__main__": main()
