import pkg_resources

from .xgboost_with_warm_start import(
    XGBClassifierWithWarmStart,
    XGBRegressorWithWarmStart,
)

__all__ = ['xgboost_with_warm_start']
__version__ = pkg_resources.get_distribution("xgboostwithwarmstart").version
