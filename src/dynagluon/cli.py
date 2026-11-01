"""Command-line interface for the DynaGluon AutoML engine."""

# External
import argparse
import importlib.metadata
import logging
from pathlib import Path
import sys

from dynagluon.engine import TabDynaGluonEngine

# Internal
from dynagluon.utils import Config, setup_logger

logger = logging.getLogger(__name__)

# Retrieve package version dynamically
try:
    __version__ = importlib.metadata.version('dynagluon')
except importlib.metadata.PackageNotFoundError:
    __version__ = '0.1.0'


def build_parser() -> argparse.ArgumentParser:
    """Construct and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog='dynagluon', description='Dynagluon AutoML Engine CLI'
    )
    parser.add_argument(
        '--version', action='version', version=f'%(prog)s {__version__}'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Enable debug logging'
    )
    parser.add_argument(
        '--env',
        default=None,
        help='Active environment to load from configuration (e.g. dev, prod)',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Train subcommand ---
    train_cmd = subparsers.add_parser('train', help='Train models')
    train_cmd.add_argument(
        '--config',
        default='config.yaml',
        help='Path to experiment YAML configuration (default: config.yaml)',
    )
    train_cmd.add_argument(
        '--data', required=True, type=Path, help='Path to training CSV data'
    )

    # --- Retrain subcommand ---
    retrain_cmd = subparsers.add_parser(
        'retrain', help='Retrain an existing model with new data'
    )
    retrain_cmd.add_argument(
        '--model-path',
        required=True,
        type=Path,
        help='Path to the existing AutogluonModels directory',
    )
    retrain_cmd.add_argument(
        '--data',
        required=True,
        type=Path,
        help='Path to new training CSV data',
    )
    retrain_cmd.add_argument(
        '--config',
        default='config.yaml',
        help='Configuration file name inside model directory (default: config.yaml)',
    )

    # --- Predict subcommand ---
    predict_cmd = subparsers.add_parser(
        'predict', help='Predict with trained model'
    )
    predict_cmd.add_argument(
        '--config',
        default='config.yaml',
        help='Path to experiment YAML configuration (default: config.yaml)',
    )
    predict_cmd.add_argument(
        '--model-path',
        required=True,
        type=Path,
        help='Path to the trained AutogluonModels directory',
    )
    predict_cmd.add_argument(
        '--data', required=True, type=Path, help='Path to evaluation CSV data'
    )
    predict_cmd.add_argument(
        '--output',
        default=Path('predictions.csv'),
        type=Path,
        help='Output path for predictions CSV',
    )

    return parser


def _run_train(args: argparse.Namespace, settings: Config) -> int:
    raw_config = Config.load(args.config, env=args.env)
    log_to_file = bool(raw_config.get('log_to_file', True))

    config = Config.load(args.config, namespace='train', env=args.env)
    config_dict = config.to_dict()
    if 'log_to_file' not in config_dict:
        config_dict['log_to_file'] = log_to_file

    save_path = str(config_dict.get('save_path', 'AutogluonModels/'))
    if log_to_file and save_path:
        setup_logger(settings, log_file=Path(save_path) / 'run.log')

    logger.info('Training configuration: %s', config_dict)
    engine = TabDynaGluonEngine(config_dict=config_dict)
    engine.train(train_data_path=str(args.data))

    # Save config into the model output folder for record-keeping
    model_save_path = engine.save_path
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    config.save(Path(model_save_path) / 'config.yaml')
    return 0


def _run_retrain(args: argparse.Namespace) -> int:
    if not args.model_path.exists():
        logger.error('Model path does not exist: %s', args.model_path)
        return 1

    config_path = args.model_path / args.config
    if not config_path.exists():
        logger.error(
            'Configuration file not found in model directory: %s',
            config_path,
        )
        return 1

    # Check for 'retrain' section first, fallback to 'train' section
    config = Config.load(config_path, namespace='retrain', env=args.env)
    if not config:
        config = Config.load(config_path, namespace='train', env=args.env)

    logger.info(
        'Retraining configuration loaded from %s: %s',
        config_path,
        config.to_dict(),
    )
    # TODO: engine = TabDynaGluonEngine.load(args.model_path)
    # TODO: engine.retrain(train_data=args.data, config=config.to_dict())
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    if not args.model_path.exists():
        logger.error('Model path does not exist: %s', args.model_path)
        return 1

    config = Config.load(args.config, namespace='predict', env=args.env)
    logger.info('Prediction configuration: %s', config.to_dict())
    engine = TabDynaGluonEngine.load(
        str(args.model_path), config_dict=config.to_dict()
    )
    engine.predict(test_data_path=str(args.data), output_path=str(args.output))
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Parse command-line arguments and run the requested workflow.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to sys.argv[1:].

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure system logging from settings.yaml
    settings = Config.load('settings.yaml', env=args.env)
    setup_logger(settings)

    # Override logging level if verbose flag is set
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug('Verbose debug logging enabled.')

    # Fast early validation of data file
    if not args.data.exists():
        logger.error('Data file does not exist: %s', args.data)
        return 1

    handlers = {
        'train': lambda: _run_train(args, settings),
        'retrain': lambda: _run_retrain(args),
        'predict': lambda: _run_predict(args),
    }
    handler = handlers.get(args.command)
    if handler:
        return handler()

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning('Operation cancelled by user. Exiting cleanly...')
        sys.exit(130)
