import numpy as np

# Compatibilidad de NumPy 2.x con librerías legadas (ej. DeepChecks)
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NaN"):
    np.NaN = np.nan

from mibici_mlops import config  # noqa: F401
