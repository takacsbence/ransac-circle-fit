try:
    from scipy import stats as _scipy_stats  # not used in v5_4, kept for compatibility
    SCIPY_AVAILABLE = True
except Exception:
    _scipy_stats = None
    SCIPY_AVAILABLE = False