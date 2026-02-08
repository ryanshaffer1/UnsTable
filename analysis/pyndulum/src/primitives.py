from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from pint import Quantity
from pint_pandas import PintArray

from src import ureg


@dataclass
class State:
    x: Quantity
    vx: Quantity
    theta: Quantity
    omega: Quantity

    @classmethod
    def history_to_dataframe(cls, history: np.ndarray, times: Quantity) -> pd.DataFrame:
        # Create dataframe with PintArray columns for state history
        df = pd.DataFrame(index=PintArray(times, dtype=times.units),
                            data={name: PintArray(data, dtype=units)
                             for (data, name, units) in zip(history,
                                                            cls.__get_variable_names(),
                                                            cls.__get_pandas_dtypes(),
                                                            strict=True,
                                                            )})
        # Convert to display units (if different from base units)
        for col, display_units in zip(df.columns, cls.get_display_units(), strict=True):
            df[col] = df[col].pint.to(display_units)
        return df

    def to_vector(self) -> list[Quantity]:
        return np.array([self.x.to_base_units().magnitude,
                         self.vx.to_base_units().magnitude,
                         self.theta.to_base_units().magnitude,
                         self.omega.to_base_units().magnitude])

    @classmethod
    def from_vector(cls, vector: list[float]) -> Self:
        return cls(x = vector[0] * ureg.meter,
                   vx = vector[1] * ureg.meter / ureg.second,
                   theta = vector[2] * ureg.radian,
                   omega = vector[3] * ureg.radian / ureg.second)

    def to_display_units(self) -> dict[str, Quantity]:
        return {name: getattr(self, name).to(display_units)
                for (name, display_units) in zip(self.__get_variable_names(),
                                                 self.get_display_units(),
                                                 strict=True)}

    @classmethod
    def get_display_units(cls) -> tuple[Quantity]:
        return (ureg.meter,
                ureg.meter / ureg.second,
                ureg.degree,
                ureg.degree / ureg.second,
                )

    def add_vector(self, vector: np.ndarray) -> Self:
        x, vx, theta, omega = self.add_vector_unitless(
            self.x, self.vx, self.theta, self.omega,
            vector[0], vector[1], vector[2], vector[3],
        )

        return State(x, vx, theta, omega)

    @ureg.wraps((ureg.meter,
                 ureg.meter/ureg.second,
                 ureg.radian,
                 ureg.radian/ureg.second,
                 ),
                (None,
                 ureg.meter,
                 ureg.meter/ureg.second,
                 ureg.radian,
                 ureg.radian/ureg.second,
                 ureg.meter,
                 ureg.meter/ureg.second,
                 ureg.radian,
                 ureg.radian/ureg.second,
                 ),
                strict=False)
    def add_vector_unitless(self, # noqa: PLR0913
                            x1: float,
                            vx1: float,
                            theta1: float,
                            omega1: float,
                            x2: float,
                            vx2: float,
                            theta2: float,
                            omega2: float,
                            ) -> tuple[float, float, float, float]:
        x = x1 + x2
        vx = vx1 + vx2
        theta = theta1 + theta2
        omega = omega1 + omega2
        return x, vx, theta, omega

    @classmethod
    def __get_variable_names(cls) -> tuple[str]:
        return ("x", "vx", "theta", "omega")

    @classmethod
    def __get_pandas_dtypes(cls) -> tuple[str]:
        return "pint[m]", "pint[m/s]", "pint[rad]", "pint[rad/s]"

@dataclass
class Input:
    u: Quantity

    @classmethod
    def history_to_dataframe(cls, history: np.ndarray, times: Quantity) -> pd.DataFrame:
        # Create dataframe with PintArray column for input history
        df = pd.DataFrame(index=PintArray(times, dtype=times.units),
                            data={"input": PintArray(history, dtype=ureg.newton)})
        # Convert to display units (if different from base units)
        df["input"] = df["input"].pint.to(ureg.newton)
        return df


class BlockPrim(ABC):
    width: Quantity
    height: Quantity

    @abstractmethod
    def get_ll_corner(self, state: State) -> tuple[float, float]:
        pass

    def get_width(self) -> Quantity:
        return self.width

    def get_height(self) -> Quantity:
        return self.height

class BeamPrim(ABC):
    length: Quantity
    thickness: Quantity

    @abstractmethod
    def get_endpoints(self, state: State) -> tuple[tuple[float, float], tuple[float, float]]:
        pass
