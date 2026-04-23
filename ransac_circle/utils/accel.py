# ---------------------------------------------------------------
# NUMBA acceleration (optional)
# ---------------------------------------------------------------
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# ---------------------------------------------------------------
# Status lines
# ---------------------------------------------------------------
def print_accel_status(quiet=False):
    if quiet:
        return
    if NUMBA_AVAILABLE:
        try:
            import numba
            print(f"Numba acceleration: ENABLED (numba {numba.__version__})")
        except Exception:
            print("Numba acceleration: ENABLED")
    else:
        print("Numba acceleration: DISABLED")
