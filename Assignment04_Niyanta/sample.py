import array
import numpy as np
import warnings
import scipy.sparse as sp
import itertools

from .base import baseestimator, classifiermixin, clone, is_classifier
from .base import multioutputmixin
from .base import metaestimatormixin, is_regressor
from .preprocessing import labelbinarizer
from .metrics.pairwise import euclidean_distances
from .utils import check_random_state
from .utils._tags import _safe_tags
from .utils.validation import _num_samples
from .utils.validation import check_is_fitted
from .utils.multiclass import (
    _check_partial_fit_first_call,
    check_classification_targets,
    _ovr_decision_function,
)
from .utils.metaestimators import _safe_split, available_if
from .utils.fixes import delayed

from joblib import parallel

__all__ = [
    "onevsrestclassifier",
    "onevsoneclassifier",
    "outputcodeclassifier",
]


def _fit_binary(estimator, x, y, classes=none):
    """fit a single binary estimator."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not none:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn(
                "label %s is present in all training examples." % str(classes[c])
            )
        estimator = _constantpredictor().fit(x, unique_y)
    else:
        estimator = clone(estimator)
        estimator.fit(x, y)
    return estimator


def _partial_fit_binary(estimator, x, y):
    """partially fit a single binary estimator."""
    estimator.partial_fit(x, y, np.array((0, 1)))
    return estimator


def _predict_binary(estimator, x):
    """make predictions using a single binary estimator."""
    if is_regressor(estimator):
        return estimator.predict(x)
    try:
        score = np.ravel(estimator.decision_function(x))
    except (attributeerror, notimplementederror):
        # probabilities of the positive class
        score = estimator.predict_proba(x)[:, 1]
    return score


def _threshold_for_binary_predict(estimator):
    """threshold for predictions from binary estimator."""
    if hasattr(estimator, "decision_function") and is_classifier(estimator):
        return 0.0
    else:
        # predict_proba threshold
        return 0.5


def _check_estimator(estimator):
    """make sure that an estimator implements the necessary methods."""
    if not hasattr(estimator, "decision_function") and not hasattr(
        estimator, "predict_proba"
    ):
        raise valueerror(
            "the base estimator should implement decision_function or predict_proba!"
        )


class _constantpredictor(baseestimator):
    def fit(self, x, y):
        check_params = dict(
            force_all_finite=false, dtype=none, ensure_2d=false, accept_sparse=true
        )
        self._validate_data(
            x, y, reset=true, validate_separately=(check_params, check_params)
        )
        self.y_ = y
        return self

    def predict(self, x):
        check_is_fitted(self)
        self._validate_data(
            x,
            force_all_finite=false,
            dtype=none,
            accept_sparse=true,
            ensure_2d=false,
            reset=false,
        )

        return np.repeat(self.y_, _num_samples(x))

    def decision_function(self, x):
        check_is_fitted(self)
        self._validate_data(
            x,
            force_all_finite=false,
            dtype=none,
            accept_sparse=true,
            ensure_2d=false,
            reset=false,
        )

        return np.repeat(self.y_, _num_samples(x))

    def predict_proba(self, x):
        check_is_fitted(self)
        self._validate_data(
            x,
            force_all_finite=false,
            dtype=none,
            accept_sparse=true,
            ensure_2d=false,
            reset=false,
        )
        y_ = self.y_.astype(np.float64)
        return np.repeat([np.hstack([1 - y_, y_])], _num_samples(x), axis=0)


def _estimators_has(attr):
    """check if self.estimator or self.estimators_[0] has attr.

    if `self.estimators_[0]` has the attr, then its safe to assume that other
    values has it too. this function is used together with `avaliable_if`.
    """
    return lambda self: (
        hasattr(self.estimator, attr)
        or (hasattr(self, "estimators_") and hasattr(self.estimators_[0], attr))
    )


class onevsrestclassifier(
    multioutputmixin, classifiermixin, metaestimatormixin, baseestimator
):
    """one-vs-the-rest (ovr) multiclass strategy.

    also known as one-vs-all, this strategy consists in fitting one classifier
    per class. for each classifier, the class is fitted against all the other
    classes. in addition to its computational efficiency (only `n_classes`
    classifiers are needed), one advantage of this approach is its
    interpretability. since each class is represented by one and one classifier
    only, it is possible to gain knowledge about the class by inspecting its
    corresponding classifier. this is the most commonly used strategy for
    multiclass classification and is a fair default choice.

    onevsrestclassifier can also be used for multilabel classification. to use
    this feature, provide an indicator matrix for the target `y` when calling
    `.fit`. in other words, the target labels should be formatted as a 2d
    binary (0/1) matrix, where [i, j] == 1 indicates the presence of label j
    in sample i. this estimator uses the binary relevance method to perform
    multilabel classification, which involves training one binary classifier
    independently for each label.

    read more in the :ref:`user guide <ovr_classification>`.

    parameters
    ----------
    estimator : estimator object
        an estimator object implementing :term:`fit` and one of
        :term:`decision_function` or :term:`predict_proba`.

    n_jobs : int, default=none
        the number of jobs to use for the computation: the `n_classes`
        one-vs-rest problems are computed in parallel.

        ``none`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. see :term:`glossary <n_jobs>`
        for more details.

        .. versionchanged:: 0.20
           `n_jobs` default changed from 1 to none

    verbose : int, default=0
        the verbosity level, if non zero, progress messages are printed.
        below 50, the output is sent to stderr. otherwise, the output is sent
        to stdout. the frequency of the messages increases with the verbosity
        level, reporting all iterations at 10. see :class:`joblib.parallel` for
        more details.

        .. versionadded:: 1.1

    attributes
    ----------
    estimators_ : list of `n_classes` estimators
        estimators used for predictions.

    classes_ : array, shape = [`n_classes`]
        class labels.

    n_classes_ : int
        number of classes.

    label_binarizer_ : labelbinarizer object
        object used to transform multiclass labels to binary labels and
        vice-versa.

    multilabel_ : boolean
        whether a onevsrestclassifier is a multilabel classifier.

    n_features_in_ : int
        number of features seen during :term:`fit`. only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        names of features seen during :term:`fit`. only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    see also
    --------
    multioutputclassifier : alternate way of extending an estimator for
        multilabel classification.
    sklearn.preprocessing.multilabelbinarizer : transform iterable of iterables
        to binary indicator matrix.

    examples
    --------
    >>> import numpy as np
    >>> from sklearn.multiclass import onevsrestclassifier
    >>> from sklearn.svm import svc
    >>> x = np.array([
    ...     [10, 10],
    ...     [8, 10],
    ...     [-5, 5.5],
    ...     [-5.4, 5.5],
    ...     [-20, -20],
    ...     [-15, -20]
    ... ])
    >>> y = np.array([0, 0, 1, 1, 2, 2])
    >>> clf = onevsrestclassifier(svc()).fit(x, y)
    >>> clf.predict([[-19, -20], [9, 9], [-5, 5]])
    array([2, 0, 1])
    """

    def __init__(self, estimator, *, n_jobs=none, verbose=0):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, x, y):
        """fit underlying estimators.

        parameters
        ----------
        x : (sparse) array-like of shape (n_samples, n_features)
            data.

        y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
            multi-class targets. an indicator matrix turns on multilabel
            classification.

        returns
        -------
        self : object
            instance of fitted estimator.
        """
        # a sparse labelbinarizer, with sparse_output=true, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        self.label_binarizer_ = labelbinarizer(sparse_output=true)
        y = self.label_binarizer_.fit_transform(y)
        y = y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in y.t)
        # in cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  see joblib issue #112.
        self.estimators_ = parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_binary)(
                self.estimator,
                x,
                column,
                classes=[
                    "not %s" % self.label_binarizer_.classes_[i],
                    self.label_binarizer_.classes_[i],
                ],
            )
            for i, column in enumerate(columns)
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    @available_if(_estimators_has("partial_fit"))
    def partial_fit(self, x, y, classes=none):
        """partially fit underlying estimators.

        should be used when memory is inefficient to train all data.
        chunks of data can be passed in several iteration.

        parameters
        ----------
        x : (sparse) array-like of shape (n_samples, n_features)
            data.

        y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
            multi-class targets. an indicator matrix turns on multilabel
            classification.

        classes : array, shape (n_classes, )
            classes across all calls to partial_fit.
            can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            this argument is only required in the first call of partial_fit
            and can be omitted in the subsequent calls.

        returns
        -------
        self : object
            instance of partially fitted estimator.
        """
        if _check_partial_fit_first_call(self, classes):
            if not hasattr(self.estimator, "partial_fit"):
                raise valueerror(
                    ("base estimator {0}, doesn't have partial_fit method").format(
                        self.estimator
                    )
                )
            self.estimators_ = [clone(self.estimator) for _ in range(self.n_classes_)]

            # a sparse labelbinarizer, with sparse_output=true, has been
            # shown to outperform or match a dense label binarizer in all
            # cases and has also resulted in less or equal memory consumption
            # in the fit_ovr function overall.
            self.label_binarizer_ = labelbinarizer(sparse_output=true)
            self.label_binarizer_.fit(self.classes_)

        if len(np.setdiff1d(y, self.classes_)):
            raise valueerror(
                (
                    "mini-batch contains {0} while classes " + "must be subset of {1}"
                ).format(np.unique(y), self.classes_)
            )

        y = self.label_binarizer_.transform(y)
        y = y.tocsc()
        columns = (col.toarray().ravel() for col in y.t)

        self.estimators_ = parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_binary)(estimator, x, column)
            for estimator, column in zip(self.estimators_, columns)
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_

        return self

    def predict(self, x):
        """predict multi-class targets using underlying estimators.

        parameters
        ----------
        x : (sparse) array-like of shape (n_samples, n_features)
            data.

        returns
        -------
        y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
            predicted multi-class targets.
        """
        check_is_fitted(self)

        n_samples = _num_samples(x)
        if self.label_binarizer_.y_type_ == "multiclass":
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            for i, e in enumerate(self.estimators_):
                pred = _predict_binary(e, x)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = i
            return self.classes_[argmaxima]
        else:
            thresh = _threshold_for_binary_predict(self.estimators_[0])
            indices = array.array("i")
            indptr = array.array("i", [0])
            for e in self.estimators_:
                indices.extend(np.where(_predict_binary(e, x) > thresh)[0])
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix(
                (data, indices, indptr), shape=(n_samples, len(self.estimators_))
            )
            return self.label_binarizer_.inverse_transform(indicator)

    @available_if(_estimators_has("predict_proba"))
    def predict_proba(self, x):
        """probability estimates.

        the returned estimates for all classes are ordered by label of classes.

        note that in the multilabel case, each sample can have any number of
        labels. this returns the marginal probability that the given sample has
        the label in question. for example, it is entirely consistent that two
        labels both have a 90% probability of applying to a given sample.

        in the single label multiclass case, the rows of the returned matrix
        sum to 1.

        parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            input data.

        returns
        -------
        t : (sparse) array-like of shape (n_samples, n_classes)
            returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self)
        # y[i, j] gives the probability that sample i has the label j.
        # in the multi-label case, these are not disjoint.
        y = np.array([e.predict_proba(x)[:, 1] for e in self.estimators_]).t

        if len(self.estimators_) == 1:
            # only one estimator, but we still want to return probabilities
            # for two classes.
            y = np.concatenate(((1 - y), y), axis=1)

        if not self.multilabel_:
            # then, probabilities should be normalized to 1.
            y /= np.sum(y, axis=1)[:, np.newaxis]
        return y

    @available_if(_estimators_has("decision_function"))
    def decision_function(self, x):
        """decision function for the onevsrestclassifier.

        return the distance of each sample from the decision boundary for each
        class. this can only be used with estimators which implement the
        `decision_function` method.

        parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            input data.

        returns
        -------
        t : array-like of shape (n_samples, n_classes) or (n_samples,) for \
            binary classification.
            result of calling `decision_function` on the final estimator.

            .. versionchanged:: 0.19
                output shape changed to ``(n_samples,)`` to conform to
                scikit-learn conventions for binary classification.
        """
        check_is_fitted(self)
        if len(self.estimators_) == 1:
            return self.estimators_[0].decision_function(x)
        return np.array(
            [est.decision_function(x).ravel() for est in self.estimators_]
        ).t

    @property
    def multilabel_(self):
        """whether this is a multilabel classifier."""
        return self.label_binarizer_.y_type_.startswith("multilabel")

    @property
    def n_classes_(self):
        """number of classes."""
        return len(self.classes_)

    def _more_tags(self):
        """indicate if wrapped estimator is using a precomputed gram matrix"""
        return {"pairwise": _safe_tags(self.estimator, key="pairwise")}


def _fit_ovo_binary(estimator, x, y, i, j):
    """fit a single binary estimator (one-vs-one)."""
    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    y_binary = np.empty(y.shape, int)
    y_binary[y == i] = 0
    y_binary[y == j] = 1
    indcond = np.arange(_num_samples(x))[cond]
    return (
        _fit_binary(
            estimator,
            _safe_split(estimator, x, none, indices=indcond)[0],
            y_binary,
            classes=[i, j],
        ),
        indcond,
    )


def _partial_fit_ovo_binary(estimator, x, y, i, j):
    """partially fit a single binary estimator(one-vs-one)."""

    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    if len(y) != 0:
        y_binary = np.zeros_like(y)
        y_binary[y == j] = 1
        return _partial_fit_binary(estimator, x[cond], y_binary)
    return estimator


class onevsoneclassifier(metaestimatormixin, classifiermixin, baseestimator):
    """one-vs-one multiclass strategy.

    this strategy consists in fitting one classifier per class pair.
    at prediction time, the class which received the most votes is selected.
    since it requires to fit `n_classes * (n_classes - 1) / 2` classifiers,
    this method is usually slower than one-vs-the-rest, due to its
    o(n_classes^2) complexity. however, this method may be advantageous for
    algorithms such as kernel algorithms which don't scale well with
    `n_samples`. this is because each individual learning problem only involves
    a small subset of the data whereas, with one-vs-the-rest, the complete
    dataset is used `n_classes` times.

 



   
