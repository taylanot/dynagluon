from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import shutil
import time

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

        # Check if model already exists and refit or extend is requested
        if self._is_trained_predictor_dir(self.save_path):
            if self.config.get('refit_if_exists', False):
                logger.info(
                    'Existing model detected at %s with refit_if_exists=True. Running refit...',
                    self.save_path,
                )
                target = self.config.get('new_save_path')
                model_to_refit = str(self.config.get('refit_model', 'best'))
                return self.refit(
                    new_save_path=str(target) if target else None,
                    model=model_to_refit,
                )

            if self.config.get('extend_if_exists', False) or self.config.get(
                'fit_extra_if_exists', False
            ):
                logger.info(
                    'Existing model detected at %s with extend_if_exists=True. Running extend...',
                    self.save_path,
                )
                target = self.config.get('new_save_path')
                return self.extend(
                    new_save_path=str(target) if target else None,
                )

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

        start_time = time.perf_counter()

        logger.info('Starting AutoGluon training...')
        self.predictor.fit(
            train_data=ag_train_data,
            **fit_kwargs,
        )

        fit_duration = time.perf_counter() - start_time
        logger.info('Training complete in %.2f seconds.', fit_duration)

        # Save run summary and leaderboard artifacts
        self._save_run_summary(
            train_data_path=train_data_path,
            num_rows=len(ag_train_data),
            duration_seconds=fit_duration,
        )

        return self.predictor

    @staticmethod
    def _is_trained_predictor_dir(path: Path | str) -> bool:
        """Check if a directory contains an existing trained AutoGluon predictor."""
        p = Path(path)
        if not p.is_dir():
            return False
        if hasattr(TabularPredictor, 'can_load'):
            try:
                return bool(TabularPredictor.can_load(str(p)))
            except (AttributeError, TypeError, OSError, ValueError) as e:
                logger.debug(
                    'TabularPredictor.can_load check failed for %s: %s', p, e
                )
        return (p / 'predictor_metadata.json').exists() or (
            p / 'models'
        ).exists()

    def extend(
        self,
        new_save_path: str | Path | None = None,
    ) -> TabularPredictor:
        """
        Extend an existing trained model with additional models or search trials.

        Parameters
        ----------
        new_save_path : str or Path, optional
            Target directory to save the extended model. If None, an isolated
            directory '<save_path>_extended_<timestamp>' is created.

        Returns
        -------
        TabularPredictor
            The updated AutoGluon predictor.

        Raises
        ------
        FileNotFoundError
            If the source model directory does not exist.
        """
        source_dir = Path(self.save_path)
        if not source_dir.exists():
            raise FileNotFoundError(
                f'Source model directory does not exist: {source_dir}'
            )

        # Resolve destination path
        if new_save_path:
            target_dir = Path(new_save_path)
        else:
            timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
            clean_base = str(self.save_path).rstrip('/\\')
            target_dir = Path(f'{clean_base}_extended_{timestamp}')

        logger.info('=' * 70)
        logger.info('Starting model extension (fit_extra)')
        logger.info('Source model directory : %s', source_dir)
        logger.info('Target run directory   : %s', target_dir)
        logger.info('=' * 70)

        # Clone artifacts into the target directory if separate from source
        if target_dir.resolve() != source_dir.resolve():
            logger.info(
                'Cloning original model artifacts from %s to %s...',
                source_dir,
                target_dir,
            )
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
            self.save_path = str(target_dir)

        # Load predictor pointing to target directory
        logger.info('Loading predictor from %s...', self.save_path)
        self.predictor = TabularPredictor.load(
            self.save_path, require_version_match=False
        )

        # Configure file logging inside the target run directory
        if self.config.get('log_to_file', True):
            setup_logger(log_file=Path(self.save_path) / 'run.log')

        # Extract hyperparameters and optional time_limit from fit_kwargs
        fit_kwargs_raw = self.config.get('autogluon_fit_kwargs', {})
        fit_kwargs = resolve_config(fit_kwargs_raw)
        extra_kwargs: dict[str, object] = {}
        hyperparameters: object = 'default'
        if isinstance(fit_kwargs, dict):
            if 'time_limit' in fit_kwargs:
                extra_kwargs['time_limit'] = fit_kwargs['time_limit']
            if fit_kwargs.get('hyperparameters'):
                hyperparameters = fit_kwargs['hyperparameters']

        start_time = time.perf_counter()
        logger.info('Executing predictor.fit_extra()...')
        self.predictor.fit_extra(
            hyperparameters,
            **extra_kwargs,
        )

        duration = time.perf_counter() - start_time
        logger.info('Extension completed in %.2f seconds.', duration)

        # Save run summary and leaderboard artifacts in target directory
        self._save_run_summary(
            train_data_path=str(self.save_path),
            num_rows=0,
            duration_seconds=duration,
        )

        logger.info(
            'Extension artifacts successfully saved to: %s',
            self.save_path,
        )
        return self.predictor

    def fit_extra(
        self,
        new_save_path: str | Path | None = None,
    ) -> TabularPredictor:
        """Alias for extend."""
        return self.extend(new_save_path=new_save_path)

    def refit(
        self,
        new_save_path: str | Path | None = None,
        model: str = 'best',
    ) -> TabularPredictor:
        """
        Refit models on 100% of data without fold splits for deployment.

        Parameters
        ----------
        new_save_path : str or Path, optional
            Target directory to save the refitted model. If None, an isolated
            directory '<save_path>_refit_<timestamp>' is created.
        model : str, default 'best'
            Model or models to refit ('best' or specific model name).

        Returns
        -------
        TabularPredictor
            The updated AutoGluon predictor.

        Raises
        ------
        FileNotFoundError
            If the source model directory does not exist.
        """
        source_dir = Path(self.save_path)
        if not source_dir.exists():
            raise FileNotFoundError(
                f'Source model directory does not exist: {source_dir}'
            )

        # Resolve destination path
        if new_save_path:
            target_dir = Path(new_save_path)
        else:
            timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
            clean_base = str(self.save_path).rstrip('/\\')
            target_dir = Path(f'{clean_base}_refit_{timestamp}')

        logger.info('=' * 70)
        logger.info('Starting model refit (refit_full)')
        logger.info('Source model directory : %s', source_dir)
        logger.info('Target run directory   : %s', target_dir)
        logger.info('Refit model target     : %s', model)
        logger.info('=' * 70)

        # Clone artifacts into the target directory if separate from source
        if target_dir.resolve() != source_dir.resolve():
            logger.info(
                'Cloning original model artifacts from %s to %s...',
                source_dir,
                target_dir,
            )
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                source_dir,
                target_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('run_summary.json'),
            )
            self.save_path = str(target_dir)

        # Remove run_summary.json if present (run_summary is only for train and extend)
        summary_file = target_dir / 'run_summary.json'
        if summary_file.exists():
            summary_file.unlink()

        # Load predictor pointing to target directory
        logger.info('Loading predictor from %s...', self.save_path)
        self.predictor = TabularPredictor.load(
            self.save_path, require_version_match=False
        )

        # Configure file logging inside the target run directory
        if self.config.get('log_to_file', True):
            setup_logger(log_file=Path(self.save_path) / 'run.log')

        start_time = time.perf_counter()
        logger.info('Executing predictor.refit_full(model=%s)...', model)
        self.predictor.refit_full(model=model, set_best_to_refit_full=True)

        duration = time.perf_counter() - start_time
        logger.info('refit_full completed in %.2f seconds.', duration)

        logger.info(
            'Refit artifacts successfully saved to: %s',
            self.save_path,
        )
        return self.predictor

    def _extract_model_hyperparameters(
        self, model_name: str
    ) -> dict[str, object]:
        """Extract model details: weights for ensembles, or hyperparameters for single/bagged models."""
        if self.predictor is None:
            return {}

        weights: dict[str, float] = {}
        try:
            m_obj = self.predictor._trainer.load_model(model_name)  # noqa: SLF001
            if hasattr(m_obj, 'get_model_weights') and callable(
                m_obj.get_model_weights
            ):
                weights = m_obj.get_model_weights()
            elif (
                hasattr(m_obj, 'model')
                and hasattr(m_obj.model, 'get_model_weights')
                and callable(m_obj.model.get_model_weights)
            ):
                weights = m_obj.model.get_model_weights()
            elif hasattr(m_obj, 'model') and hasattr(m_obj.model, 'weights'):
                weights = m_obj.model.weights
            elif hasattr(m_obj, 'model') and hasattr(m_obj.model, 'get_info'):
                info = m_obj.model.get_info()
                if (
                    isinstance(info, dict)
                    and 'weights' in info
                    and isinstance(info['weights'], dict)
                ):
                    weights = info['weights']
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as e:
            logger.debug(
                'Could not inspect model weights for %s: %s', model_name, e
            )

        if weights:
            active_weights = {
                str(k): float(v) for k, v in weights.items() if float(v) > 0
            }
            return {
                'type': 'ensemble',
                'model_name': model_name,
                'ensemble_weights': active_weights,
            }

        # For base or bagged models, use predictor.model_hyperparameters
        try:
            hps = self.predictor.model_hyperparameters(model=model_name)
            hps_dict = hps if isinstance(hps, dict) else {}
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as e:
            logger.debug(
                'Could not get hyperparameters for %s: %s', model_name, e
            )
            hps_dict = {}

        return {
            'type': 'single_model',
            'model_name': model_name,
            'hyperparameters': hps_dict,
        }

    def _save_run_summary(
        self, train_data_path: str, num_rows: int, duration_seconds: float
    ) -> None:
        """Export run summary JSON and leaderboard parquet for experiment tracking."""
        if self.predictor is None:
            return

        save_dir = Path(self.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            leaderboard_df = self.predictor.leaderboard(silent=True)
            leaderboard_path = save_dir / 'leaderboard.parquet'
            leaderboard_df.to_parquet(leaderboard_path, index=False)
            logger.info('Leaderboard saved to %s', leaderboard_path)

            best_model = str(self.predictor.model_best)
            best_score_row = leaderboard_df.loc[
                leaderboard_df['model'] == best_model
            ]
            best_score = (
                float(best_score_row['score_val'].values[0])
                if not best_score_row.empty
                and pd.notna(best_score_row['score_val'].values[0])
                else None
            )

            eval_metric = (
                self.predictor.eval_metric.name
                if hasattr(self.predictor, 'eval_metric')
                else 'unknown'
            )

            # Extract details for top 3 models
            top_3_rows = leaderboard_df.head(3)
            top_3_models_info: list[dict[str, object]] = []
            for rank, (_, row) in enumerate(top_3_rows.iterrows(), start=1):
                m_name = str(row['model'])
                m_score = (
                    float(row['score_val'])
                    if pd.notna(row['score_val'])
                    else None
                )
                m_fit_time = (
                    float(row['fit_time'])
                    if 'fit_time' in row and pd.notna(row['fit_time'])
                    else None
                )
                m_details = self._extract_model_hyperparameters(m_name)
                top_3_models_info.append(
                    {
                        'rank': rank,
                        'model_name': m_name,
                        'validation_score': m_score,
                        'fit_time_seconds': m_fit_time,
                        'model_details': m_details,
                    }
                )

            best_model_details = self._extract_model_hyperparameters(
                best_model
            )

            summary = {
                'timestamp': datetime.now(UTC).isoformat(),
                'target_column': str(self.target_column),
                'problem_type': str(self.predictor.problem_type),
                'eval_metric': eval_metric,
                'train_data_path': str(train_data_path),
                'train_rows_count': num_rows,
                'total_fit_duration_seconds': round(duration_seconds, 2),
                'total_models_trained': len(leaderboard_df),
                'best_model': best_model,
                'best_validation_score': best_score,
                'best_model_hyperparameters': best_model_details,
                'top_3_models': top_3_models_info,
            }

            summary_path = save_dir / 'run_summary.json'
            with summary_path.open('w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info('Run summary saved to %s', summary_path)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            AttributeError,
        ) as e:
            logger.warning('Failed to save run summary artifacts: %s', e)

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
