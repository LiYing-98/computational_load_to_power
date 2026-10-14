"""Algorithm-isolated nested GroupCV with reconstructed-total model selection."""
import importlib.metadata
import json
import platform
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from step7.scripts.common import ROOT, SEED, TARGET, assert_feature_boundary

LOCAL_PACKAGES = ROOT / "step5/.packages"
if LOCAL_PACKAGES.exists() and str(LOCAL_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGES))


class WarmStartElasticNet(ElasticNet):
    """Initialize at the ridge part of the same penalized objective.

    This changes the numerical starting point, not alpha/l1_ratio or the
    optimization objective.  The ridge penalty equals n*alpha*(1-l1_ratio).
    """
    def fit(self, X, y, sample_weight=None, check_input=True):
        ridge = Ridge(alpha=len(y)*self.alpha*(1-self.l1_ratio),fit_intercept=self.fit_intercept)
        ridge.fit(X,y,sample_weight=sample_weight)
        self.coef_ = ridge.coef_.copy()
        self.intercept_ = ridge.intercept_
        self.warm_start = True
        return super().fit(X,y,sample_weight=sample_weight,check_input=check_input)


def versions():
    values = {"python":platform.python_version()}
    for package in ["numpy","pandas","scikit-learn","xgboost","catboost"]:
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = "unavailable"
    return values


def target_values(frame, strategy):
    values = frame[TARGET].to_numpy(float)
    return values / frame.x_hardware_num_gpus.to_numpy(float) if strategy == "G-perGPU" else values


def reconstruct_total(frame, prediction, strategy):
    return np.asarray(prediction,float) * frame.x_hardware_num_gpus.to_numpy(float) if strategy == "G-perGPU" else np.asarray(prediction,float)


def selection_mae(frame, prediction, strategy):
    return float(np.abs(reconstruct_total(frame,prediction,strategy) - frame[TARGET].to_numpy(float)).mean())


def metrics(y, predicted, num_gpus):
    y, pred, n = np.asarray(y,float), np.asarray(predicted,float), np.asarray(num_gpus,float)
    residual = pred-y
    slope, intercept = np.polyfit(y,pred,1) if len(y)>1 and np.ptp(y)>0 else (np.nan,np.nan)
    variance = ((y-y.mean())**2).sum()
    return {"mae":float(np.abs(residual).mean()),"rmse":float(np.sqrt(np.mean(residual**2))),"r2":float(1-(residual**2).sum()/variance) if variance else np.nan,"mdape":float(np.median(np.abs(residual)/np.abs(y))*100),"smape":float(np.mean(2*np.abs(residual)/(np.abs(pred)+np.abs(y)))*100),"signed_bias":float(residual.mean()),"per_gpu_mae":float(np.mean(np.abs(residual)/n)),"per_gpu_rmse":float(np.sqrt(np.mean((residual/n)**2))),"per_gpu_signed_bias":float(np.mean(residual/n)),"calibration_slope":float(slope),"calibration_intercept":float(intercept),"negative_predictions":int((pred<0).sum())}


def estimator(algorithm, candidate, bundle):
    if algorithm == "DummyMean":
        model = DummyRegressor(strategy="mean")
    elif algorithm == "Ridge":
        model = Ridge(**candidate)
    elif algorithm == "ElasticNet":
        model = WarmStartElasticNet(**candidate, max_iter=500000, tol=1e-4, random_state=SEED)
    elif algorithm in {"RandomForest","ExtraTrees"}:
        cls = RandomForestRegressor if algorithm == "RandomForest" else ExtraTreesRegressor
        model = cls(**candidate, criterion="absolute_error",random_state=SEED,n_jobs=1)
    elif algorithm == "HistGradientBoosting":
        model = HistGradientBoostingRegressor(**candidate,loss="absolute_error",random_state=SEED)
    elif algorithm == "XGBoost":
        from xgboost import XGBRegressor
        model = XGBRegressor(**candidate,objective="reg:absoluteerror",random_state=SEED,n_jobs=1,verbosity=0)
    elif algorithm == "CatBoost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**candidate,loss_function="MAE",random_seed=SEED,thread_count=1,verbose=False,allow_writing_files=False)
    else:
        raise ValueError(algorithm)
    steps = [("impute",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True))]
    if algorithm in {"Ridge","ElasticNet"}:
        steps.append(("scale",StandardScaler()))
    preprocessor = ColumnTransformer([
        ("num",Pipeline(steps),bundle["numeric"]),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent",keep_empty_features=True)),("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),bundle["categorical"]),
    ],remainder="drop")
    return Pipeline([("preprocess",preprocessor),("model",model)])


def fit_outer(train, test, algorithm, bundle, candidates, strategy):
    columns = bundle["numeric"]+bundle["categorical"]
    assert_feature_boundary(columns,task_sensitivity="x_protocol_task" in bundle["categorical"])
    train = train.sort_values("run_key").reset_index(drop=True)
    test = test.sort_values("run_key").reset_index(drop=True)
    if set(train.diag_model_id) & set(test.diag_model_id):
        raise ValueError("model ID crosses outer train/test")
    inner = list(GroupKFold(n_splits=min(3,train.diag_model_id.nunique())).split(train,groups=train.diag_model_id))
    scores, errors = [], []
    for candidate in candidates:
        total_error, row_count = 0., 0
        try:
            for tr,va in inner:
                local_train,local_val = train.iloc[tr],train.iloc[va]
                model = estimator(algorithm,candidate,bundle)
                with threadpool_limits(limits=1), warnings.catch_warnings():
                    warnings.simplefilter("error",ConvergenceWarning)
                    model.fit(local_train[columns],target_values(local_train,strategy))
                    pred = model.predict(local_val[columns])
                total_error += selection_mae(local_val,pred,strategy)*len(local_val)
                row_count += len(local_val)
            scores.append(total_error/row_count)
            errors.append(None)
        except Exception as exc:
            scores.append(float("inf"))
            errors.append(f"{type(exc).__name__}: {exc}")
    if not np.isfinite(scores).any():
        raise ValueError(f"all candidates failed: {errors}")
    selected = int(np.argmin(scores))
    model = estimator(algorithm,candidates[selected],bundle)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        model.fit(train[columns],target_values(train,strategy))
        output = reconstruct_total(test,model.predict(test[columns]),strategy)
    metadata = {"algorithm":algorithm,"target_strategy":strategy,"selection_metric":"row_weighted_reconstructed_total_mae","selected_index":selected,"selected_hyperparameters":candidates[selected],"inner_scores":[float(x) if np.isfinite(x) else None for x in scores],"candidate_failures":errors,"train_models":sorted(train.diag_model_id.unique()),"test_models":sorted(test.diag_model_id.unique()),"inner_groups":[{"train":sorted(train.iloc[a].diag_model_id.unique()),"validation":sorted(train.iloc[b].diag_model_id.unique())} for a,b in inner]}
    return output, metadata, model
