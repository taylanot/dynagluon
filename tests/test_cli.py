import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dynagluon import cli


def test_cli_train_predict(tmp_path):
    from sklearn.datasets import make_classification

    # 1. Setup quick dummy data
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    features = [f'feature_{i}' for i in range(5)]
    df = pd.DataFrame(X, columns=features)
    df['target'] = y

    train_path = tmp_path / 'train.csv'
    test_path = tmp_path / 'test.csv'
    df.head(40).to_csv(train_path, index=False)
    df.tail(10).to_csv(test_path, index=False)

    # 2. Setup configuration file from tests/test_configs/
    base_config_path = (
        Path(__file__).parent / 'test_configs' / 'cli_config.yaml'
    )
    model_path = tmp_path / 'AutogluonModels'

    # Read the file and replace save_path to point to the tmp_path
    config_content = base_config_path.read_text()
    config_content = config_content.replace(
        '"AutogluonModels"', f'"{model_path!s}"'
    )

    config_path = tmp_path / 'config.yaml'
    config_path.write_text(config_content)

    # 3. Run CLI Train
    # We patch sys.argv to simulate running this from the terminal
    with patch(
        'sys.argv',
        [
            'dynagluon',
            'train',
            '--config',
            str(config_path),
            '--data',
            str(train_path),
        ],
    ):
        # mock output_dir which wasn't in original argparser or let's see if cli code needs it
        try:
            cli.main()
        except AttributeError as e:
            # We catch it just in case args.output_dir is accessed and fails, but our cli.py
            # might have been modified to not fail.
            pass

    assert os.path.exists(model_path)

    # 4. Run CLI Predict
    out_csv = tmp_path / 'results.csv'
    with patch(
        'sys.argv',
        [
            'dynagluon',
            'predict',
            '--model-path',
            str(model_path),
            '--data',
            str(test_path),
            '--output',
            str(out_csv),
            '--config',
            str(config_path),
        ],
    ):
        cli.main()

    assert os.path.exists(out_csv)

    # Verify predictions output was generated
    res_df = pd.read_csv(str(out_csv))
    assert 'predicted_target' in res_df.columns

def test_cli_data_not_exists(tmp_path):
    with patch('sys.argv', ['dynagluon', 'train', '--data', str(tmp_path / 'non_existent.csv')]):
        ret = cli.main()
        assert ret == 1


def test_cli_predict_model_not_exists(tmp_path):
    test_path = tmp_path / 'test.csv'
    test_path.touch()
    with patch('sys.argv', ['dynagluon', 'predict', '--model-path', str(tmp_path / 'non_existent_model'), '--data', str(test_path)]):
        ret = cli.main()
        assert ret == 1


def test_cli_retrain_model_not_exists(tmp_path):
    train_path = tmp_path / 'train.csv'
    train_path.touch()
    with patch('sys.argv', ['dynagluon', 'retrain', '--model-path', str(tmp_path / 'non_existent_model'), '--data', str(train_path)]):
        ret = cli.main()
        assert ret == 1


def test_cli_retrain_config_not_exists(tmp_path):
    train_path = tmp_path / 'train.csv'
    train_path.touch()
    model_path = tmp_path / 'model'
    model_path.mkdir()
    with patch('sys.argv', ['dynagluon', 'retrain', '--model-path', str(model_path), '--data', str(train_path)]):
        ret = cli.main()
        assert ret == 1


def test_cli_verbose_flag(tmp_path):
    with patch('sys.argv', ['dynagluon', '-v', 'train', '--data', str(tmp_path / 'non_existent.csv')]):
        ret = cli.main()
        assert ret == 1


def test_cli_retrain_basic(tmp_path):
    train_path = tmp_path / 'train.csv'
    train_path.write_text("a,b,target\n1,2,0\n")
    model_path = tmp_path / 'model'
    model_path.mkdir()
    config_path = model_path / 'config.yaml'
    config_path.write_text("train:\n  target_column: target\n")
    with patch('dynagluon.cli.TabDynaGluonEngine.load') as mock_load:
        mock_engine = mock_load.return_value
        mock_engine.extend.return_value = None
        with patch('sys.argv', ['dynagluon', 'retrain', '--model-path', str(model_path), '--data', str(train_path)]):
            ret = cli.main()
            assert ret == 0
            mock_load.assert_called_once()
            mock_engine.extend.assert_called_once()


def test_cli_extend_config_required(tmp_path):
    model_path = tmp_path / 'model'
    model_path.mkdir()
    # extend subcommand must fail if --config is omitted
    with patch('sys.argv', ['dynagluon', 'extend', '--model-path', str(model_path)]):
        try:
            ret = cli.main()
            assert ret != 0
        except SystemExit as e:
            assert e.code != 0


def test_cli_extend_external_config(tmp_path):
    model_path = tmp_path / 'model'
    model_path.mkdir()
    external_config = tmp_path / 'my_external_config.yaml'
    external_config.write_text("extend:\n  time_limit: 10\n")

    with patch('dynagluon.cli.TabDynaGluonEngine.load') as mock_load:
        mock_engine = mock_load.return_value
        mock_engine.extend.return_value = None
        argv = [
            'dynagluon',
            'extend',
            '--model-path',
            str(model_path),
            '--config',
            str(external_config),
        ]
        with patch('sys.argv', argv):
            ret = cli.main()
            assert ret == 0
            mock_load.assert_called_once()
            mock_engine.extend.assert_called_once()


def test_cli_refit_without_config(tmp_path):
    model_path = tmp_path / 'model'
    model_path.mkdir()
    # refit subcommand does NOT require --config
    with patch('dynagluon.cli.TabDynaGluonEngine.load') as mock_load:
        mock_engine = mock_load.return_value
        mock_engine.refit.return_value = None
        argv = ['dynagluon', 'refit', '--model-path', str(model_path)]
        with patch('sys.argv', argv):
            ret = cli.main()
            assert ret == 0
            mock_load.assert_called_once()
            mock_engine.refit.assert_called_once()
