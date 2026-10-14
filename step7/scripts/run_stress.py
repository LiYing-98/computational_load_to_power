"""Fixed-choice post-hoc lineage/architecture diagnostics, never tuning input."""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from step7.scripts.common import STEP7,TARGET,read_json,sha256,write_json
from step7.scripts.freeze_protocol import verify_protocol
from step7.scripts.modeling import estimator,target_values,reconstruct_total,metrics,versions
from step7.scripts.run_matrix import checkpoint_key,valid_checkpoint


def make_stress_manifests(frame):
    identity=["run_key","diag_model_id","diag_parent_lineage","diag_architecture_class","x_protocol_task"]
    outputs={}
    for scheme,column in [("parent_lineage","diag_parent_lineage"),("architecture","diag_architecture_class")]:
        chunks=[]
        for fold,group in enumerate(sorted(frame[column].unique())):
            assignment=frame.loc[frame[column].eq(group),identity].copy()
            assignment["held_out_group"]=group
            assignment["stress_fold"]=fold
            assignment["post_hoc_only"]=True
            assignment["scheme"]=scheme
            chunks.append(assignment)
        outputs[scheme]=pd.concat(chunks,ignore_index=True)
    chunks=[]
    fold=0
    for task,task_frame in frame.groupby("x_protocol_task"):
        if task_frame.diag_parent_lineage.nunique()<3:
            continue
        for lineage in sorted(task_frame.diag_parent_lineage.unique()):
            assignment=task_frame.loc[task_frame.diag_parent_lineage.eq(lineage),identity].copy()
            assignment["held_out_group"]=lineage
            assignment["held_constant_task"]=task
            assignment["stress_fold"]=fold
            assignment["post_hoc_only"]=True
            assignment["scheme"]="within_task_lineage"
            chunks.append(assignment)
            fold+=1
    outputs["within_task_lineage"]=pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame(columns=identity+["held_out_group","held_constant_task","stress_fold","post_hoc_only","scheme"])
    return outputs


def stress_dependencies():
    return {"modeling.py":sha256(STEP7/"scripts/modeling.py"),"common.py":sha256(STEP7/"scripts/common.py"),"protocol":sha256(STEP7/"config/protocol_freeze.json"),"source_freeze":sha256(STEP7/"config/source_freeze_phase5.json")}


def main():
    verify_protocol()
    lock_path=STEP7/"config/development_candidates_lock.json"
    lock=read_json(lock_path)
    frame=pd.read_parquet(STEP7/"analysis/phase5_master_table.parquet").sort_values("run_key").reset_index(drop=True)
    hierarchy=read_json(STEP7/"config/feature_hierarchy_phase5.json")["levels"]
    summaries=[]
    all_predictions=[]
    for level in ["L0-Basic","L1-Deployment","L2-StaticCore"]:
        choice=lock["choices"][level]
        metadata=read_json(STEP7/"models/development_candidates"/f"{level}.json")
        bundle=hierarchy[level]
        support=frame.loc[frame[bundle["eligibility"]]].copy()
        manifests=make_stress_manifests(support)
        for scheme,manifest in manifests.items():
            manifest_path=STEP7/"splits"/f"post_hoc_{level}_{scheme}.csv"
            manifest.to_csv(manifest_path,index=False)
            fingerprint=checkpoint_key({"candidate_lock":sha256(lock_path),"metadata":metadata,"manifest":sha256(manifest_path),"code":sha256(__file__),"versions":versions(),"dependencies":stress_dependencies()})
            for fold in sorted(manifest.stress_fold.unique()):
                assigned=manifest.loc[manifest.stress_fold.eq(fold)]
                held_group=assigned.held_out_group.iloc[0]
                task=assigned.x_protocol_task.iloc[0] if scheme=="within_task_lineage" else None
                test_keys=set(assigned.run_key)
                test=support.loc[support.run_key.isin(test_keys)].copy()
                train=support.loc[~support.run_key.isin(test_keys)].copy()
                if task is not None:
                    train=train.loc[train.x_protocol_task.eq(task)]
                if set(train.diag_model_id)&set(test.diag_model_id):
                    raise ValueError("stress model identity leaked")
                path=STEP7/"models/checkpoints"/fingerprint/f"{fold}.parquet"
                if valid_checkpoint(path,fingerprint,test_keys):
                    result=pd.read_parquet(path)
                else:
                    model=estimator(choice["algorithm"],metadata["selected_hyperparameters"],bundle)
                    columns=bundle["numeric"]+bundle["categorical"]
                    with threadpool_limits(limits=1),warnings.catch_warnings():
                        warnings.simplefilter("error",ConvergenceWarning)
                        model.fit(train[columns],target_values(train,choice["target_strategy"]))
                        prediction=reconstruct_total(test,model.predict(test[columns]),choice["target_strategy"])
                    result=test[["run_key","diag_model_id","diag_parent_lineage","diag_subfamily","diag_architecture_class","x_protocol_task","x_hardware_num_gpus","x_hardware_gpu_model",TARGET]].copy()
                    result["prediction"]=prediction
                    result["residual"]=prediction-result[TARGET].to_numpy(float)
                    result["stress_scheme"]=scheme
                    result["stress_fold"]=int(fold)
                    result["feature_level"]=level
                    result["held_out_group"]=held_group
                    result["checkpoint_fingerprint"]=fingerprint
                    result["post_hoc_only"]=True
                    path.parent.mkdir(parents=True,exist_ok=True)
                    result.to_parquet(path,index=False)
                summary={"feature_level":level,"algorithm":choice["algorithm"],"target_strategy":choice["target_strategy"],"scheme":scheme,"held_out_group":held_group,"task":task,"post_hoc_only":True,"n_rows":len(result),"n_models":result.diag_model_id.nunique(),"train_rows":len(train),"train_models":train.diag_model_id.nunique(),"gpu_composition":json.dumps(result.x_hardware_gpu_model.value_counts().to_dict()),"task_composition":json.dumps(result.x_protocol_task.value_counts().to_dict()),"task_lineage_confounded":bool(support.loc[support.diag_parent_lineage.eq(held_group),"x_protocol_task"].nunique()==1) if scheme!="architecture" else None,**metrics(result[TARGET],result.prediction,result.x_hardware_num_gpus)}
                summaries.append(summary)
                all_predictions.append(result)
                print(f"post-hoc complete: {level} {scheme} {held_group}",flush=True)
    pd.DataFrame(summaries).to_csv(STEP7/"reports/generalization_stress.csv",index=False)
    pd.concat(all_predictions,ignore_index=True).to_parquet(STEP7/"analysis/post_hoc_stress_predictions.parquet",index=False)
    write_json(STEP7/"analysis/stress_receipt.json",{"status":"complete","candidate_lock_sha256":sha256(lock_path),"n_stress_cells":len(summaries),"prediction_rows":sum(len(x) for x in all_predictions),"selection_feedback":False})


if __name__ == "__main__":
    main()
