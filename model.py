import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import ParameterGrid


def lgb_cv(train_df,num_folds,not_feat_cols=["id"],label_col_name='label'):
    """
    二分类lgb内置cv调参代码
    :param train_df:
    :param num_folds:
    :param not_feat_cols:
    :param label_col_name:
    :return:
    """
    param_grid = {
        "num_leaves":[15,20,25],
        "max_depth":[4,5,6],
        "learning_rate":[0.02,0.05],
        "reg_alpha":[0,0.01,0.05,0.1],
        "reg_lambda":[0,0.01,0.05]
    }
    params_list = list(ParameterGrid(param_grid))
    feats = [col for col in train_df.columns if col not in not_feat_cols]
    lgb_train = lgb.Dataset(train_df[feats],train_df[label_col_name])
    cv_results = []
    for param in params_list:
        hyperparams = {
            "objective":"binary",
            "boosting_type":"gbdt",
            "n_jobs":-1,
            "random_state":2020,
            "n_estimators":5000,
            "num_leaves":param["num_leaves"],
            "max_depth":param["max_depth"],
            "learning_rate":param["learning_rate"],
            "reg_alpha":param["reg_alpha"],
            "reg_lambda":param["reg_lambda"]
        }
        validation_summary = lgb.cv(hyperparams,lgb_train,nfold=num_folds,metrics=['auc'],early_stopping_rounds=50,verbose_eval=None)
        cv_results.append((param,validation_summary["auc-mean"][-1]))
    return cv_results

# cv_results = lgb_cv(train_df,4)
# best_param = sorted(cv_results,key=lambda x:-x[1])[0][0]


def kfold_lightgbm(train_df,test_df,num_folds,param,not_feat_cols=["id"],label_col_name='label',stratified=False,debug=True):
    """
    二分类k折lgb
    :param train_df:
    :param test_df:
    :param num_folds:
    :param param: lgb训练参数
    :param not_feat_cols:
    :param label_col_name:
    :param stratified:
    :param debug:
    :return:
    """
    print("Starting Lightgbm. Train Shape: {}, Test Shape: {}".format(train_df.shape,test_df.shape))
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=2020)
    else:
        folds = KFold(n_splits=num_folds,shuffle=True,random_state=2020)
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [col for col in train_df.columns if col not in not_feat_cols]

    for n_fold,(train_idx,valid_idx) in enumerate(folds.split(train_df[feats],train_df[label_col_name])):
        train_x, train_y = train_df[feats].iloc[train_idx],train_df[label_col_name].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx],train_df[label_col_name].iloc[valid_idx]

        clf = lgb.LGBMClassifier(**param)
        clf.fit(train_x,train_y,eval_set=[(train_x,train_y),(valid_x,valid_y)],eval_metric='auc',verbose=100,early_stopping_rounds=50)
        oof_preds[valid_idx] = clf.predict_proba(valid_x,num_iteration=clf.best_iteration_)[:,1]
        sub_preds += clf.predict_proba(test_df[feats],num_iteration=clf.best_iteration_)[:,1]/folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold+1
        feature_importance_df = pd.concat([feature_importance_df,fold_importance_df],axis=0)

    test_df['pred'] = sub_preds
    test_df = test_df.sort_values(by="pred",ascending=False)
    if debug:
        top300 = test_df.iloc[:300,:][label_col_name].sum()
        print("top300:{}, top300 hit ratio:{}".format(top300,top300/300.0))
        print("测试集样本总数：{}, 正样本总数：{}".format(test_df.shape[0],test_df[label_col_name].sum()))
    feature_importance_df = feature_importance_df.groupby("feature")["importance"].sum().sort_values(ascending=False)
    return test_df,feature_importance_df

