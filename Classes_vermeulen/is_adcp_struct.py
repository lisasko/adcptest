from typing import Any, Sequence
import numpy as np

## OPTIONNEL peut-être
## à voir si on l'utilise 
## à voir si on créé un .py avec toutes les fonctions utilitaires comme celle-ci

def is_adcp_struct(inp: Any, flags: str = "ev") -> Sequence[bool]:
    if inp is None:
        return False
    # If single ADCP-like object, consider True if it has some key attributes
    def check_one(x):
        if not hasattr(x, "raw") and not hasattr(x, "water_velocity"):
            return False
        # minimal fields for 'e' (ensemble) and 'v' indicate presence of arrays
        if "e" in flags and not (hasattr(x, "time") or hasattr(x, "nensembles")):
            return False
        return True
    try:
        # array-like
        return np.array([check_one(v) for v in inp], dtype=bool)
    except TypeError:
        return check_one(inp)