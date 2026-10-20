import argparse
import logging

from dynaconf import Dynaconf

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('dynagluon.cli')


def load_settings(config_path: str) -> Dynaconf:
    logger.info('Loading configuration from %s', config_path)
    return Dynaconf(settings_files=[config_path], environments=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Dynagluon AutoML Engine CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # TRAIN
    train_cmd = subparsers.add_parser('train', help='Train models')
    train_cmd.add_argument(
        '--config',
        default='settings.yaml',
        help='Path to Dynaconf yaml (default: settings.yaml)',
    )
    train_cmd.add_argument(
        '--data', required=True, help='Path to training CSV data'
    )

    # PREDICT
    predict_cmd = subparsers.add_parser(
        'predict', help='Predict with trained model'
    )
    predict_cmd.add_argument(
        '--model-path',
        required=True,
        help='Path to the trained AutogluonModels directory',
    )
    predict_cmd.add_argument(
        '--data', required=True, help='Path to evaluation CSV data'
    )
    predict_cmd.add_argument(
        '--output',
        default='predictions.csv',
        help='Output path for predictions CSV',
    )

    args = parser.parse_args()

    if args.command == 'train':
        settings = load_settings(args.config)
        config_dict = settings.get('train', {'a': 1, 'b': 0.2})
        logger.info(config_dict)

    if args.command == 'predict':
        settings = load_settings(args.config)
        config_dict = settings.get('predict', {'a': 1, 'b': 0.2})
        logger.info(config_dict)


if __name__ == '__main__':
    main()
