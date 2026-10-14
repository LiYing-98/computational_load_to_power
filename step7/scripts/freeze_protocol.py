"""Freeze deterministic group splits and finite candidates before training."""
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, ParameterSampler

from step7.scripts.common import ROOT, STEP7, SEED, read_json, sha256, verify_sources, write_json

ALGORITHMS = ["DummyMean", "Ridge", "ElasticNet", "HistGradientBoosting", "RandomForest", "ExtraTrees", "XGBoost", "CatBoost"]


def spaces():
    forest = {"n_estimators":[300,600], "max_depth":[None,8,12,16,24], "min_samples_leaf":[1,2,4,8], "min_samples_split":[2,4,8], "max_features":[.4,.6,.8,1.]}
    return {
        "DummyMean": {},
        "Ridge": {"alpha":np.logspace(-3,4,24).tolist()},
        "ElasticNet": {"alpha":np.logspace(-4,2,24).tolist(), "l1_ratio":[.05,.2,.5,.8,.95]},
        "RandomForest":forest,
        "ExtraTrees":forest,
        "HistGradientBoosting":{"max_iter":[200,400,700],"learning_rate":[.02,.05,.1],"max_leaf_nodes":[15,31,63],"min_samples_leaf":[5,10,20,30],"l2_regularization":[0.,1.,3.,10.]},
        "XGBoost":{"n_estimators":[200,500,1000],"max_depth":[2,3,4,6],"learning_rate":[.02,.05,.1],"min_child_weight":[1,3,5,10],"subsample":[.7,.85,1.],"colsample_bytree":[.7,.85,1.],"reg_alpha":[0.,.1,1.],"reg_lambda":[1.,3.,10.]},
        "CatBoost":{"iterations":[300,600,1000],"depth":[4,5,6,8],"learning_rate":[.02,.05,.1],"l2_leaf_reg":[3.,5.,10.,20.],"random_strength":[0.,.5,1.]},
    }


def verify_protocol():
    protocol = read_json(STEP7 / "config/protocol_freeze.json")
    for item in protocol["artifacts"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen protocol changed: {item['path']}")
    verify_sources()
    return protocol


def main():
    if (STEP7 / "config/protocol_freeze.json").exists():
        verify_protocol()
        print("protocol already frozen; no file rewritten")
        return
    verify_sources()
    table = pd.read_parquet(STEP7 / "analysis/phase5_master_table.parquet")
    levels = read_json(STEP7 / "config/feature_hierarchy_phase5.json")["levels"]
    definitions = spaces()
    sampled = {name: [{}] if name == "DummyMean" else list(ParameterSampler(definitions[name], n_iter=8, random_state=SEED)) for name in ALGORITHMS}
    write_json(STEP7 / "config/algorithm_search_spaces_phase5.json", {"search_budget":8,"default_24_reduced_before_training":True,"seed":SEED,"algorithms":definitions,"linear_loss_note":"Ridge/ElasticNet are penalized least-square baselines; selection always total MAE","tree_loss":"absolute_error/MAE"})
    write_json(STEP7 / "config/sampled_hyperparameter_candidates.json", {"seed":SEED,"algorithms":sampled})
    write_json(STEP7 / "config/target_strategies_phase5.json", {"T-total":{"train":"aggregate GPU watts","selection":"row-weighted total watts MAE"},"G-perGPU":{"train":"aggregate watts / num_gpus","restore":"prediction_per_gpu * num_gpus","selection":"row-weighted restored total watts MAE","meaning":"aggregate mean GPU power, not individual-card measurement"}})
    cohorts, support_for = {}, {}
    for level, definition in levels.items():
        keys = tuple(sorted(table.loc[table[definition["eligibility"]],"run_key"]))
        existing = next((name for name, values in cohorts.items() if values == keys), None)
        if existing is None:
            existing = {"L0-Basic":"full", "L2-StaticCore":"static", "L4-CompactPhysics":"compact", "L3E-Effective":"effective"}.get(level, "support_"+str(len(cohorts)))
            cohorts[existing] = keys
        support_for[level] = existing
    split_dir = STEP7 / "splits"
    split_dir.mkdir(exist_ok=True)
    manifests = []
    for name, keys in cohorts.items():
        rows = table.loc[table.run_key.isin(keys)].sort_values("run_key").reset_index(drop=True)
        if rows.diag_model_id.nunique() < 6:
            raise ValueError(f"insufficient models for stable nested 5-fold support: {name}")
        outer = np.full(len(rows), -1)
        for fold, (_, test) in enumerate(GroupKFold(n_splits=5).split(rows, groups=rows.diag_model_id)):
            outer[test] = fold
        manifest = rows[["run_key","diag_model_id","diag_parent_lineage","diag_architecture_class","x_protocol_task"]].copy()
        manifest["outer_fold"] = outer
        manifest.to_csv(split_dir / f"outer_model_folds_{name}.csv", index=False)
        manifests.append({"cohort":name,"n_rows":len(rows),"n_models":rows.diag_model_id.nunique(),"sha256":sha256(split_dir / f"outer_model_folds_{name}.csv")})
    experiment_pairs = [(level, support_for[level], "sensitivity" if definition["sensitivity_only"] else "main") for level, definition in levels.items()]
    increments = [("L0-Basic","L0A-ActiveAware"),("L0A-ActiveAware","L1-Deployment"),("L1-Deployment","L2-StaticCore"),("L2-StaticCore","L3-Workload"),("L3-Workload","L4-CompactPhysics"),("L2-StaticCore","L2X-StaticExtended"),("L3-Workload","L3E-Effective"),("L4-CompactPhysics","L4B5-OldB5Sensitivity"),("L1-Deployment","L1-TaskSensitivity")]
    if "L1T-TokenizerFreeTextProxy" in levels:
        increments.append(("L1-Deployment","L1T-TokenizerFreeTextProxy"))
    pairs = []
    for lower, higher in increments:
        cohort = support_for[higher]
        if not any(level == lower and c == cohort for level,c,_ in experiment_pairs):
            experiment_pairs.append((lower,cohort,"matched_support_reference"))
        pairs.append({"lower":lower,"higher":higher,"cohort":cohort})
    experiments = []
    for (level,cohort,role), algorithm, target in itertools.product(experiment_pairs, ALGORITHMS, ["T-total","G-perGPU"]):
        experiments.append({"experiment_id":f"{level}__{cohort}__{algorithm}__{target}","feature_level":level,"cohort":cohort,"algorithm":algorithm,"target_strategy":target,"analysis_role":role})
    write_json(STEP7 / "config/experiment_plan.json", {"cohorts":list(cohorts),"experiments":experiments,"adjacent_pairs":pairs})
    write_json(split_dir / "split_manifest_phase5.json", {"seed":SEED,"outer_folds":5,"inner_folds":3,"group":"diag_model_id","cohorts":manifests})
    pd.DataFrame([{"cohort":name,"run_key":key} for name, keys in cohorts.items() for key in keys]).to_csv(STEP7 / "analysis/matched_support_manifest.csv", index=False)
    paths = ["analysis/phase5_master_table.parquet", "config/feature_hierarchy_phase5.json", "config/feature_formulas_phase5.json", "config/algorithm_search_spaces_phase5.json", "config/sampled_hyperparameter_candidates.json", "config/target_strategies_phase5.json", "config/experiment_plan.json", "analysis/matched_support_manifest.csv"] + [f"splits/outer_model_folds_{name}.csv" for name in cohorts]
    write_json(STEP7 / "config/protocol_freeze.json", {"schema_version":"1.0","frozen_before_training":True,"artifacts":[{"path":"step7/"+p,"sha256":sha256(STEP7 / p)} for p in paths]})
    print(json.dumps({"cohorts":manifests,"experiments":len(experiments),"budget_per_algorithm":8}, indent=2))


if __name__ == "__main__":
    main()
