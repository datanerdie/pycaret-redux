"""PyCaret Redux — A modernized, low-code classification library."""

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

from pycaret_redux.experiment import ClassificationExperiment  # noqa: E402

__version__ = "0.1.0"
__all__ = ["ClassificationExperiment"]
