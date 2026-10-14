"""Lock/final-fit tiered engineering candidates before post-hoc stresses."""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from step7.scripts.common import STEP7, TARGET, read_json, sha256, write_json
from step7.scripts.freeze_protocol import verify_protocol
from step7.scripts.modeling import estimator, target_values, selection_mae, versions


def choose_level(matrix, level):
    eligible=matrix.loc[matrix.feature_level.eq(level)&matrix.analysis_role.eq("main")]
    if eligible.empty or eligible.cohort.nunique()!=1:
        raise ValueError("candidate selection needs one main cohort per level")
    return eligible.sort_values(["mae","rmse","algorithm","target_strategy"]).iloc[0].to_dict()


def fit_final(frame,algorithm,bundle,candidates,strategy):
    frame=frame.sort_values("run_key").reset_index(drop=True)
    splits=list(GroupKFold(n_splits=3).split(frame,groups=frame.diag_model_id))
    columns=bundle["numeric"]+bundle["categorical"]
    scores,failures=[],[]
    for candidate in candidates:
        errors,count=0.,0
        try:
            for tr,va in splits:
                model=estimator(algorithm,candidate,bundle)
                with threadpool_limits(limits=1),warnings.catch_warnings():
                    warnings.simplefilter("error",ConvergenceWarning)
                    model.fit(frame.iloc[tr][columns],target_values(frame.iloc[tr],strategy))
                    pred=model.predict(frame.iloc[va][columns])
                errors+=selection_mae(frame.iloc[va],pred,strategy)*len(va)
                count+=len(va)
            scores.append(errors/count)
            failures.append(None)
        except Exception as exc:
            scores.append(float("inf"))
            failures.append(f"{type(exc).__name__}: {exc}")
    if not np.isfinite(scores).any():
        raise ValueError("all final group-CV candidates failed")
    index=int(np.argmin(scores))
    model=estimator(algorithm,candidates[index],bundle)
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        model.fit(frame[columns],target_values(frame,strategy))
    return model,{"selected_index":index,"selected_hyperparameters":candidates[index],"group_cv_total_mae":[float(s) if np.isfinite(s) else None for s in scores],"candidate_failures":failures}


def main():
    verify_protocol()
    status=read_json(STEP7/"analysis/matrix_progress.json")
    if status["status"]!="complete":
        raise ValueError("cannot lock development candidates before complete matrix")
    matrix=pd.read_csv(STEP7/"reports/main_model_matrix.csv")
    frame=pd.read_parquet(STEP7/"analysis/phase5_master_table.parquet")
    hierarchy=read_json(STEP7/"config/feature_hierarchy_phase5.json")["levels"]
    sampled=read_json(STEP7/"config/sampled_hyperparameter_candidates.json")["algorithms"]
    lock_path=STEP7/"config/development_candidates_lock.json"
    choices={level:choose_level(matrix,level) for level in ["L0-Basic","L1-Deployment","L2-StaticCore","L3-Workload"]}
    # Serialize only stable scientific choices/selection metrics, no stress results.
    lock={"selection":"observed lowest main nested OOF total MAE; RMSE breaks ties; best-of-level winner bias acknowledged","development_candidate_not_external_validated":True,"source_matrix_sha256":sha256(STEP7/"reports/main_model_matrix.csv"),"choices":choices}
    if lock_path.exists():
        saved=read_json(lock_path)
        if saved!=lock:
            amendment_path=STEP7/"config/l2x_semantic_amendment.json"
            if not amendment_path.exists() or saved["choices"]!=lock["choices"] or saved["source_matrix_sha256"]!=read_json(amendment_path)["original_matrix_sha256"]:
                raise ValueError("candidate lock changed; do not retune after stress")
            # L2X is sensitivity only; original locked selection evidence is archived.
            lock=saved
    else:
        write_json(lock_path,lock)
    directory=STEP7/"models/development_candidates"
    directory.mkdir(parents=True,exist_ok=True)
    for level,choice in choices.items():
        path=directory/f"{level}.joblib"
        metadata_path=directory/f"{level}.json"
        if path.exists() and metadata_path.exists():
            metadata=read_json(metadata_path)
            if metadata["model_sha256"]==sha256(path) and metadata["lock_sha256"]==sha256(lock_path):
                print(f"candidate already frozen: {level}",flush=True)
                continue
            raise ValueError("existing development candidate hash mismatch")
        bundle=hierarchy[level]
        support=frame.loc[frame[bundle["eligibility"]]]
        model,fit_info=fit_final(support,choice["algorithm"],bundle,sampled[choice["algorithm"]],choice["target_strategy"])
        joblib.dump(model,path,compress=3)
        write_json(metadata_path,{"level":level,"algorithm":choice["algorithm"],"target_strategy":choice["target_strategy"],"feature_columns":bundle["numeric"]+bundle["categorical"],"n_rows":len(support),"n_models":support.diag_model_id.nunique(),"development_candidate_not_external_validated":True,"model_sha256":sha256(path),"lock_sha256":sha256(lock_path),"versions":versions(),**fit_info})
        print(f"development candidate fitted: {level}",flush=True)


if __name__ == "__main__":
    main()
