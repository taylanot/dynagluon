import os
from pathlib import Path

import pandas as pd
import pytest

from dynagluon.engine import TabDynaGluonEngine
from dynagluon.utils import Config


@pytest.fixture
def dummy_data(tmp_path):
    from sklearn.datasets import make_classification

    # Generate a binary classification dataset
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    features = [f'feature_{i}' for i in range(5)]
    df = pd.DataFrame(X, columns=features)
    df['target'] = y

    train_path = tmp_path / 'train.csv'
    test_path = tmp_path / 'test.csv'

    df.head(80).to_csv(train_path, index=False)
    df.tail(20).to_csv(test_path, index=False)

    return str(train_path), str(test_path)


@pytest.mark.parametrize("config_filename", ["tune_mlp.yaml", "tune_gbm.yaml"])
def test_hyperparameter_tuning(dummy_data, tmp_path, config_filename):
    train_csv, test_csv = dummy_data

    # Load configuration
    config_file = Path(__file__).parent / 'test_configs' / config_filename
    cfg = Config.load(config_file, namespace='train')

    # Update save_path to use tmp_path
    cfg.data['save_path'] = str(tmp_path / f'AutogluonModels_{config_filename}')

    # Train
    engine = TabDynaGluonEngine(config_dict=cfg.data)
    engine.train(train_csv)

    assert os.path.exists(cfg.data['save_path'])

    # Predict
    out_csv = os.path.join(cfg.data['save_path'], 'preds.csv')
    preds_df = engine.predict(test_csv, output_path=out_csv)

    assert os.path.exists(out_csv)
    assert f'predicted_{cfg.data["target_column"]}' in preds_df.columns
    assert len(preds_df) == 20
