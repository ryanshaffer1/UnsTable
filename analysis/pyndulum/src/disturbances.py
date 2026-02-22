from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pint import Quantity
from pint_pandas import PintArray

from src import ureg
from src.primitives import State

rng = np.random.default_rng(seed=42)

class Disturbance(ABC):
    def __init__(self,
                 start_time: Quantity = 0.0*ureg.second,
                 end_time: Quantity = np.inf*ureg.second) -> None:
        self.start_time = start_time
        self.end_time = end_time

    @abstractmethod
    def apply(self, state: State, time: float) -> np.ndarray:
        pass

    @classmethod
    def history_to_dataframe(cls, history: np.ndarray, times: Quantity) -> pd.DataFrame:
        # Create dataframe with PintArray columns for disturbance history
        df = pd.DataFrame(index=PintArray(times, dtype=times.units),
                          data={name: PintArray(data, dtype=units)
                            for (data, name, units) in zip(history,
                                            [f"w_{var}" for var in State.get_variable_names()],
                                            State.get_pandas_dtypes(),
                                            strict=True,
                                            )})
        # Convert to display units (if different from base units)
        for col, display_units in zip(df.columns, State.get_display_units(), strict=True):
            df[col] = df[col].pint.to(display_units)
        return df



class GaussianDisturbance(Disturbance):
    def __init__(self,
                 mean: np.ndarray | None = None,
                 std_dev: np.ndarray | None = None,
                 **kwargs: dict,
                 ) -> None:
        # Defaults for optional inputs
        if mean is None:
            mean = np.zeros(4)
        if std_dev is None:
            std_dev = np.array([0, 0.05, 0, 0.05])

        super().__init__(**kwargs)
        self.mean = mean
        self.std_dev = std_dev

    def apply(self, state: State, time: float) -> np.ndarray:
        # Do not apply disturbance if current time is outside of disturbance window
        if time < self.start_time or time > self.end_time:
            return state

        # Generate noise for each state variable and add to the state
        noise = rng.normal(self.mean, self.std_dev)
        return noise
