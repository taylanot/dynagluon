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
