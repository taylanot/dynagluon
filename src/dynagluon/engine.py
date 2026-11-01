"""TabDynaGluon engine for training and inference."""

import logging
from pathlib import Path

from autogluon.tabular import TabularPredictor
import pandas as pd

from dynagluon.utils import resolve_config, setup_logger

logger = logging.getLogger(__name__)


class TabDynaGluonEngine:
    """Core engine handling training and inference of AutoGluon models."""

    def __init__(self, config_dict: dict[str, object] | None = None) -> None:
        """
        Initialize the TabDynaGluonEngine.

        Parameters
        ----------
        config_dict : dict of str to object, optional
            Configuration containing global settings and autogluon_fit_kwargs.
        """
        if not (config_dict is None or isinstance(config_dict, dict)):
            raise TypeError(
                'config_dict must be a dictionary or None (for loading)'
            )
        self.config = config_dict or {}

        self.predictor: TabularPredictor | None = None
        self.save_path = str(self.config.get('save_path', 'AutogluonModels/'))
        self.target_column = self.config.get('target_column')

    @staticmethod
    def _load_data(path: str) -> pd.DataFrame:
        ext = Path(path).suffix.lower()
        readers = {
            '.csv': pd.read_csv,
            '.parquet': pd.read_parquet,
            '.json': pd.read_json,
        }
        if ext not in readers:
            raise ValueError(f'Unsupported file extension: {ext}')
        return readers[ext](path)

    @staticmethod
    def _save_data(df: pd.DataFrame, path: str) -> None:
        ext = Path(path).suffix.lower()
        writers = {
            '.csv': df.to_csv,
            '.parquet': df.to_parquet,
            '.json': df.to_json,
        }
        if ext not in writers:
            raise ValueError(f'Unsupported file extension: {ext}')

        if ext == '.csv':
            writers[ext](path, index=False)
        else:
            writers[ext](path)

    def train(self, train_data_path: str) -> TabularPredictor:
        """
        Train an AutoGluon predictor using data from train_data_path.

        Uses Polars for fast loading and PyArrow as an efficient bridge.

        Parameters
        ----------
        train_data_path : str
            Path to the training data CSV.

        Returns
        -------
        TabularPredictor
            The trained AutoGluon predictor.

        Raises
        ------
        ValueError
            If 'target_column' is not defined in the configuration.
        """
        if not self.target_column:
            raise ValueError("Configuration must define 'target_column'.")

        # Configure file logging to save_path/run.log if enabled
        if self.config.get('log_to_file', True) and self.save_path:
            setup_logger(log_file=Path(self.save_path) / 'run.log')

        logger.info('Loading data from %s...', train_data_path)

        # 1. Data load via Pandas (supports csv, parquet, json)
        ag_train_data = self._load_data(train_data_path)

        # 3. Resolve dynamic config objects (e.g. search spaces)
        fit_kwargs_raw = self.config.get('autogluon_fit_kwargs', {})
        fit_kwargs = resolve_config(fit_kwargs_raw)

        if not isinstance(fit_kwargs, dict):
            fit_kwargs = {}

        # 4. Initialize Predictor
        self.predictor = TabularPredictor(
            label=str(self.target_column),
            path=self.save_path,
        )

        logger.info('Starting AutoGluon training...')
        self.predictor.fit(
            train_data=ag_train_data,
            **fit_kwargs,
        )

        logger.info('Training complete.')
        return self.predictor

    @classmethod
    def load(
        cls, path: str, config_dict: dict[str, object] | None = None
    ) -> 'TabDynaGluonEngine':
        """
        Load a trained engine for inference.

        Parameters
        ----------
        path : str
            Path to the saved TabularPredictor model.
        config_dict : dict of str to object, optional
            Configuration dictionary for inference.

        Returns
        -------
        DynaGluonEngine
            Instantiated engine with loaded predictor.
        """
        config = config_dict or {}
        config['save_path'] = path
        engine = cls(config_dict=config)
        engine.predictor = TabularPredictor.load(path)
        return engine

    def predict(
        self, test_data_path: str, output_path: str = 'predictions.csv'
    ) -> pd.DataFrame:
        """
        Run inference using the trained Predictor.

        Returns and saves the predictions.

        Parameters
        ----------
        test_data_path : str
            Path to the test data CSV/Parquet/JSON.
        output_path : str, default 'predictions.csv'
            Path where the predictions will be saved.

        Returns
        -------
        pd.DataFrame
            The test data with predictions appended.

        Raises
        ------
        RuntimeError
            If the model has not been trained or loaded yet.
        """
        if self.predictor is None:
            raise RuntimeError('Model not trained or loaded. Cannot predict.')

        logger.info(
            'Loading test data from %s for inference...', test_data_path
        )

        # 1. Read Test Data with Pandas
        ag_test_data = self._load_data(test_data_path)

        # 2. Extract specific model name if provided in config
        model_to_use = self.config.get('model', None)
        if model_to_use:
            logger.info('Using specific model from config: %s', model_to_use)
        else:
            logger.info(
                'No specific model provided, using default (best) model.'
            )

        # 3. Predict (Returns a Pandas Series or DataFrame)
        logger.info('Running predictions...')
        predictions_pd = self.predictor.predict(
            ag_test_data, model=model_to_use
        )

        # 3. Attach the predictions as a new column to the input data
        pred_column_name = f'predicted_{self.predictor.label}'
        ag_test_data[pred_column_name] = predictions_pd

        # 4. Save output
        self._save_data(ag_test_data, output_path)
        logger.info('Predictions successfully saved to %s', output_path)

        return ag_test_data
