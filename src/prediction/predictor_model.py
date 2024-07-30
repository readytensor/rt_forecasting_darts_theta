import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from darts.models.forecasting.theta import Theta
from darts.utils.utils import SeasonalityMode
from darts import TimeSeries
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError
from joblib import Parallel, delayed
from multiprocessing import cpu_count

warnings.filterwarnings("ignore")

PREDICTOR_FILE_NAME = "predictor.joblib"


class Forecaster:
    """A wrapper class for the Theta Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "Theta Forecaster"

    def __init__(
        self,
        theta: int = 2,
        seasonality_period: Optional[int] = None,
        season_mode: SeasonalityMode = SeasonalityMode.MULTIPLICATIVE,
    ):
        """Construct a new Theta Forecaster

        Args:
            theta (int): Value of the theta parameter. Defaults to 2. Cannot be set to 0.
                    If theta = 1, then the theta method restricts to a simple exponential
                    smoothing (SES)
            seasonality_period (Optional[int]): User-defined seasonality period.
                    If not set, will be tentatively inferred from the training series upon
                    calling fit().
            season_mode (SeasonalityMode): Type of seasonality.
                    Either SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.ADDITIVE or
                    SeasonalityMode.NONE. Defaults to SeasonalityMode.MULTIPLICATIVE.
        """
        self.theta = theta
        self.seasonality_period = seasonality_period
        self.season_mode = season_mode
        self._is_trained = False
        self.models = {}
        self.data_schema = None

    def map_frequency(self, frequency: str) -> str:
        frequency = frequency.lower().split("frequency.")[1]
        mapping = {
            "yearly": "Y",
            "quarterly": "Q",
            "monthly": "M",
            "weekly": "W",
            "daily": "D",
            "hourly": "H",
            "minutely": "min",
            "secondly": "S",
        }
        return mapping.get(frequency, "S")

    def fit(self, history: pd.DataFrame, data_schema: ForecastingSchema) -> None:
        np.random.seed(0)
        history.set_index(
            data_schema.id_col, inplace=True
        )  # Set index for faster filtering
        groups_by_ids = history.groupby(level=0)  # Group by the index

        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).reset_index(drop=True) for id_ in all_ids
        ]

        def fit_model(id_, series):
            return id_, self._fit_on_series(history=series, data_schema=data_schema)

        n_jobs = max(1, cpu_count() - 2)
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_model)(id_, series) for id_, series in zip(all_ids, all_series)
        )
        self.models = dict(results)

        self.all_ids = all_ids
        self._is_trained = True
        self.data_schema = data_schema

    def _fit_on_series(self, history: pd.DataFrame, data_schema: ForecastingSchema):
        model = Theta(
            theta=self.theta,
            season_mode=self.season_mode,
            seasonality_period=self.seasonality_period,
        )

        series = TimeSeries.from_dataframe(history, value_cols=data_schema.target)
        model.fit(series)

        return model

    def predict(self, test_data: pd.DataFrame, prediction_col_name: str) -> np.ndarray:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        groups_by_ids = test_data.groupby(self.data_schema.id_col)
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=self.data_schema.id_col)
            for id_ in self.all_ids
        ]
        # forecast one series at a time
        all_forecasts = []
        forecast_length = len(all_series[0])
        for id_, series_df in zip(self.all_ids, all_series):
            forecast = self._predict_on_series(key_and_future_df=(id_, series_df, forecast_length))
            forecast.insert(0, self.data_schema.id_col, id_)
            all_forecasts.append(forecast)

        # concatenate all series' forecasts into a single dataframe
        all_forecasts = pd.concat(all_forecasts, axis=0, ignore_index=True)

        all_forecasts.rename(
            columns={self.data_schema.target: prediction_col_name}, inplace=True
        )
        return all_forecasts

    def _predict_on_series(self, key_and_future_df):
        """Make forecast on given individual series of data"""
        key, future_df, forecast_length = key_and_future_df
        model = self.models[key]
        if self.models.get(key) is not None:
            forecast = model.predict(forecast_length)
            forecast_df = forecast.pd_dataframe()
            forecast = forecast_df[self.data_schema.target]
            future_df[self.data_schema.target] = forecast.values
        else:
            # no model found - key wasnt found in history, so cant forecast for it.
            future_df = None
        return future_df

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    model = Forecaster(
        **hyperparameters,
    )
    model.fit(history=history, data_schema=data_schema)
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)
