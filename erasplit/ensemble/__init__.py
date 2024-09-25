"""
The :mod:`erasplit.ensemble` module includes ensemble-based methods for
classification, regression and anomaly detection.
"""
from ._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingRegressor,
    EraHistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)

__all__ = [
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "EraHistGradientBoostingRegressor",
]
