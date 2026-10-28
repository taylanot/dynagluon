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


@pytest.fixture
def dummy_config(tmp_path):
    # Load configuration from the external test folder
    config_file = Path(__file__).parent / 'test_configs' / 'engine_config.yaml'
    cfg = Config.load(config_file, namespace='train')

    # Update save_path to use tmp_path so we don't pollute the workspace during tests
    cfg.data['save_path'] = str(tmp_path / 'AutogluonModels')
    return cfg.data


def test_engine_train_predict(dummy_data, dummy_config):
    train_csv, test_csv = dummy_data

    # 1. Test Train
    engine = TabDynaGluonEngine(config_dict=dummy_config)
    engine.train(train_csv)

    assert os.path.exists(dummy_config['save_path'])

    # 2. Test Predict
    out_csv = os.path.join(dummy_config['save_path'], 'preds.csv')
    preds_df = engine.predict(test_csv, output_path=out_csv)

    assert os.path.exists(out_csv)

    # Check predictions attach correctly as a new column
    assert f'predicted_{dummy_config["target_column"]}' in preds_df.columns
    assert len(preds_df) == 20


def test_engine_load(dummy_data, dummy_config):
    train_csv, test_csv = dummy_data
    engine = TabDynaGluonEngine(config_dict=dummy_config)
    engine.train(train_csv)

    # 3. Test Load functionality for Inference
    loaded_engine = TabDynaGluonEngine.load(dummy_config['save_path'])
    out_csv = os.path.join(dummy_config['save_path'], 'preds_loaded.csv')
    preds_df = loaded_engine.predict(test_csv, output_path=out_csv)

    assert os.path.exists(out_csv)
    assert len(preds_df) == 20
