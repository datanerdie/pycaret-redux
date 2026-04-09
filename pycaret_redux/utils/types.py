"""Type aliases used across pycaret_redux."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Data inputs accepted by setup()
DATAFRAME_LIKE = pd.DataFrame | np.ndarray | list[list[Any]]
SEQUENCE_LIKE = list[Any] | np.ndarray | pd.Series
TARGET_LIKE = int | str | SEQUENCE_LIKE
