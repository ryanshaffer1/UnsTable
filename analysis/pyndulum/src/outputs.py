import logging

import numpy as np
import pandas as pd

from src import ureg

logger = logging.getLogger("log")

@ureg.wraps("=A",("=A","=A"), strict=False)
def extreme_value_unitless(data: np.ndarray, reference_point: float) -> float:
    return max(abs(data - reference_point))

def record_outputs(output_df: pd.DataFrame,
                   step_variable: str,
                   *args: tuple,
                   history_stats: bool = True,
                   response_metrics: bool = True,
                   log: bool = True) -> dict[str, float]:
    # Initialize outputs
    stats = {}
    metrics = {}

    # Optionally record the history statistics
    if history_stats:
        if log:
            logger.info("--- History statistics: ---")
        stats = record_history_stats(output_df, *args, log=log)

    # Optionally record the step response metrics
    if response_metrics:
        if log:
            logger.info("--- Step response metrics: ---")
        metrics = record_response_metrics(output_df, step_variable=step_variable, log=log)

    return {**stats, **metrics}

def record_history_stats(output_df: pd.DataFrame, *args: tuple, log: bool=True) -> dict[str, float]:
    # Write history statistics
    stats = {}
    for name, history in output_df.items():
        statistics = deviation_statistics(history, 0.0)
        for stat, val in statistics.items():
            if log:
                logger.info(f"{name.upper()} {stat} deviation: {val:0.3~P}") # noqa: G004
            stats[f"{name}_{stat}"] = val

    return stats

def record_response_metrics(output_df: pd.DataFrame,
                            step_variable: str,
                            *args: tuple,
                            log: bool=True,
                            ) -> dict[str, float]:
    # Write response metrics
    series = output_df[step_variable]
    target_value = 0
    initial_value = series.iloc[0]
    metrics = response_metrics(series, target_value, initial_value)
    for stat, val in metrics.items():
        if log:
            logger.info(f"{step_variable} {stat}: {val:0.3~P}") # noqa: G004

    return metrics


def calc_deviation(series: pd.Series, reference_point: float) -> pd.Series:
    return abs(series - reference_point)

def map_to_unit_response(series: pd.Series, target_value: float, initial_value: float) -> pd.Series:
    return (series - initial_value) / (target_value - initial_value)

def deviation_statistics(series: pd.Series, reference_point: float) -> dict[str, float]:
    deviation_series = calc_deviation(series, reference_point)
    mean_value = deviation_series.mean()
    median_value, percentile_95 = np.percentile(deviation_series, [50, 95])
    extreme_value = max(deviation_series)
    return {
        "mean": mean_value,
        "median": median_value,
        "95th_pct": percentile_95,
        "extreme": extreme_value,
    }

def response_metrics(series: pd.Series,
                     target_value: float,
                     initial_value: float,
                     ) -> dict[str, float]:
    # Response metric definitions
    rise_time_threshold = 0.8
    settling_time_threshold = 0.1

    # Calculate response metrics
    response_series = map_to_unit_response(series, target_value, initial_value)
    rise_time = series.index[response_series > rise_time_threshold].min()
    settling_time = series.index[abs(response_series - 1) > settling_time_threshold].max()
    overshoot = series.iloc[response_series.argmax()]
    return {
        "rise_time": rise_time,
        "settling_time": settling_time,
        "overshoot": overshoot,
    }
