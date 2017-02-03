# coding: utf-8
# pylint: disable=too-many-arguments, too-many-locals, invalid-name, fixme, E0012, R0912
"""Scikit-Learn Wrapper interface for XGBoost."""
from __future__ import absolute_import

import numpy as np
from xgboost import XGBRegressor
from xgboost.core import Booster, DMatrix, XGBoostError
from xgboost.training import train

# Do not use class names on scikit-learn directly.
# Re-define the classes on .compat to guarantee the behavior without scikit-learn
from xgboost.compat import (SKLEARN_INSTALLED, XGBModelBase,
                            XGBClassifierBase, XGBRegressorBase,
                            XGBLabelEncoder)
from xgboost.sklearn import _objective_decorator, XGBModel


class XGBRegressorWithWarmStart(XGBRegressor):

    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective="reg:linear",
                 nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None, warm_start=False):
        super(XGBRegressorWithWarmStart, self).__init__(
            max_depth, learning_rate, n_estimators,
            silent, objective,
            nthread, gamma, min_child_weight, max_delta_step,
            subsample, colsample_bytree, colsample_bylevel,
            reg_alpha, reg_lambda, scale_pos_weight,
            base_score, seed, missing)
        self.warm_start = warm_start
        self.n_trained_estimators = 0

    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        # pylint: disable=missing-docstring,invalid-name,attribute-defined-outside-init
        """
        Fit the gradient boosting model

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        sample_weight : array_like
            Weight for each instance
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as a validation set for
            early-stopping
        eval_metric : str, callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.md. If callable, a custom evaluation metric. The call
            signature is func(y_predicted, y_true) where y_true will be a
            DMatrix object such that you may need to call the get_label
            method. It must return a str, value pair where the str is a name
            for the evaluation and value is the value of the evaluation
            function. This objective is always minimized.
        early_stopping_rounds : int
            Activates early stopping. Validation error needs to decrease at
            least every <early_stopping_rounds> round(s) to continue training.
            Requires at least one item in evals.  If there's more than one,
            will use the last. Returns the model from the last iteration
            (not the best one). If early stopping occurs, the model will
            have three additional fields: bst.best_score, bst.best_iteration
            and bst.best_ntree_limit.
            (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
            and/or num_class appears in the parameters)
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.
        """
        if sample_weight is not None:
            trainDmatrix = DMatrix(X, label=y, weight=sample_weight, missing=self.missing)
        else:
            trainDmatrix = DMatrix(X, label=y, missing=self.missing)

        evals_result = {}
        if eval_set is not None:
            evals = list(DMatrix(x[0], label=x[1], missing=self.missing) for x in eval_set)
            evals = list(zip(evals, ["validation_{}".format(i) for i in
                                     range(len(evals))]))
        else:
            evals = ()

        params = self.get_xgb_params()

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            params["objective"] = "reg:linear"
        else:
            obj = None

        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                params.update({'eval_metric': eval_metric})

        if self.warm_start:
            n_estimators = self.n_estimators - self.n_trained_estimators
            self._Booster = train(params, trainDmatrix,
                                  n_estimators, evals=evals,
                                  early_stopping_rounds=early_stopping_rounds,
                                  evals_result=evals_result, obj=obj, feval=feval,
                                  verbose_eval=verbose, xgb_model=self._Booster)
        else:
            self._Booster = train(params, trainDmatrix,
                                  self.n_estimators, evals=evals,
                                  early_stopping_rounds=early_stopping_rounds,
                                  evals_result=evals_result, obj=obj, feval=feval,
                                  verbose_eval=verbose)
        self.n_trained_estimators = self.n_estimators

        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result_ = evals_result

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
            self.best_ntree_limit = self._Booster.best_ntree_limit
        return self

    @property
    def feature_importances_(self):
        """
        Returns
        -------
        feature_importances_ : array of shape = [n_features]

        """
        b = self.booster()
        fs = b.get_fscore()
        all_features = [fs.get(f, 0.) for f in b.feature_names]
        all_features = np.array(all_features, dtype=np.float32)
        return all_features / all_features.sum()


class XGBClassifierWithWarmStart(XGBModel, XGBClassifierBase):
    # pylint: disable=missing-docstring,too-many-arguments,invalid-name
    __doc__ = """Implementation of the scikit-learn API for XGBoost classification.

    """ + '\n'.join(XGBModel.__doc__.split('\n')[2:])

    def __init__(self, max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic",
                 nthread=-1, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None, warm_start=False):
        super(XGBClassifierWithWarmStart, self).__init__(
            max_depth, learning_rate, n_estimators, silent, objective,
            nthread, gamma, min_child_weight, max_delta_step, subsample,
            colsample_bytree, colsample_bylevel, reg_alpha, reg_lambda,
            scale_pos_weight, base_score, seed, missing)
        self.warm_start = warm_start
        self.n_trained_estimators = 0

    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        # pylint: disable = attribute-defined-outside-init,arguments-differ
        """
        Fit gradient boosting classifier

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        sample_weight : array_like
            Weight for each instance
        eval_set : list, optional
            A list of (X, y) pairs to use as a validation set for
            early-stopping
        eval_metric : str, callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.md. If callable, a custom evaluation metric. The call
            signature is func(y_predicted, y_true) where y_true will be a
            DMatrix object such that you may need to call the get_label
            method. It must return a str, value pair where the str is a name
            for the evaluation and value is the value of the evaluation
            function. This objective is always minimized.
        early_stopping_rounds : int, optional
            Activates early stopping. Validation error needs to decrease at
            least every <early_stopping_rounds> round(s) to continue training.
            Requires at least one item in evals.  If there's more than one,
            will use the last. Returns the model from the last iteration
            (not the best one). If early stopping occurs, the model will
            have three additional fields: bst.best_score, bst.best_iteration
            and bst.best_ntree_limit.
            (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
            and/or num_class appears in the parameters)
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.
        """
        evals_result = {}
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        params = self.get_xgb_params()

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            # Use default value. Is it really not used ?
            params["objective"] = "binary:logistic"
        else:
            obj = None

        if self.n_classes_ > 2:
            # Switch to using a multiclass objective in the underlying XGB instance
            params["objective"] = "multi:softprob"
            params['num_class'] = self.n_classes_

        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                params.update({"eval_metric": eval_metric})

        self._le = XGBLabelEncoder().fit(y)
        training_labels = self._le.transform(y)

        if eval_set is not None:
            # TODO: use sample_weight if given?
            evals = list(
                DMatrix(x[0], label=self._le.transform(x[1]), missing=self.missing)
                for x in eval_set
            )
            nevals = len(evals)
            eval_names = ["validation_{}".format(i) for i in range(nevals)]
            evals = list(zip(evals, eval_names))
        else:
            evals = ()

        self._features_count = X.shape[1]

        if sample_weight is not None:
            trainDmatrix = DMatrix(X, label=training_labels, weight=sample_weight,
                                   missing=self.missing)
        else:
            trainDmatrix = DMatrix(X, label=training_labels, missing=self.missing)

        if self.warm_start:
            n_estimators = self.n_estimators - self.n_trained_estimators
            self._Booster = train(params, trainDmatrix,
                                  n_estimators, evals=evals,
                                  early_stopping_rounds=early_stopping_rounds,
                                  evals_result=evals_result, obj=obj, feval=feval,
                                  verbose_eval=verbose, xgb_model=self._Booster)
        else:
            self._Booster = train(params, trainDmatrix,
                                  self.n_estimators, evals=evals,
                                  early_stopping_rounds=early_stopping_rounds,
                                  evals_result=evals_result, obj=obj, feval=feval,
                                  verbose_eval=verbose)
        self.n_trained_estimators = self.n_estimators

        self.objective = params["objective"]

        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result_ = evals_result

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
            self.best_ntree_limit = self._Booster.best_ntree_limit
        return self

    def predict(self, data, output_margin=False, ntree_limit=0):
        test_dmatrix = DMatrix(data, missing=self.missing)
        class_probs = self.booster().predict(test_dmatrix,
                                             output_margin=output_margin,
                                             ntree_limit=ntree_limit)
        if len(class_probs.shape) > 1:
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            column_indexes = np.repeat(0, class_probs.shape[0])
            column_indexes[class_probs > 0.5] = 1
        return self._le.inverse_transform(column_indexes)

    def predict_proba(self, data, output_margin=False, ntree_limit=0):
        test_dmatrix = DMatrix(data, missing=self.missing)
        class_probs = self.booster().predict(test_dmatrix,
                                             output_margin=output_margin,
                                             ntree_limit=ntree_limit)
        if self.objective == "multi:softprob":
            return class_probs
        else:
            classone_probs = class_probs
            classzero_probs = 1.0 - classone_probs
            return np.vstack((classzero_probs, classone_probs)).transpose()

    def evals_result(self):
        """Return the evaluation results.

        If eval_set is passed to the `fit` function, you can call evals_result() to
        get evaluation results for all passed eval_sets. When eval_metric is also
        passed to the `fit` function, the evals_result will contain the eval_metrics
        passed to the `fit` function

        Returns
        -------
        evals_result : dictionary

        Example
        -------
        param_dist = {'objective':'binary:logistic', 'n_estimators':2}

        clf = xgb.XGBClassifier(**param_dist)

        clf.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_metric='logloss',
                verbose=True)

        evals_result = clf.evals_result()

        The variable evals_result will contain:
        {'validation_0': {'logloss': ['0.604835', '0.531479']},
         'validation_1': {'logloss': ['0.41965', '0.17686']}}
        """
        if self.evals_result_:
            evals_result = self.evals_result_
        else:
            raise XGBoostError('No results.')

        return evals_result

    @property
    def feature_importances_(self):
        """
        Returns
        -------
        feature_importances_ : array of shape = [n_features]

        """
        b = self.booster()
        fs = b.get_fscore()
        all_features = [fs.get(f, 0.) for f in b.feature_names]
        all_features = np.array(all_features, dtype=np.float32)
        return all_features / all_features.sum()
