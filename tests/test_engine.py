import json
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
    log_file = os.path.join(dummy_config['save_path'], 'run.log')
    assert os.path.exists(log_file)
    assert 'Loading data from' in Path(log_file).read_text(encoding='utf-8')

    summary_file = os.path.join(dummy_config['save_path'], 'run_summary.json')
    leaderboard_file = os.path.join(dummy_config['save_path'], 'leaderboard.parquet')
    assert os.path.exists(summary_file)
    assert os.path.exists(leaderboard_file)

    with open(summary_file, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    assert 'best_model' in summary_data
    assert 'best_model_hyperparameters' in summary_data
    assert 'top_3_models' in summary_data
    assert summary_data['total_models_trained'] > 0
    assert len(summary_data['top_3_models']) > 0

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

def test_engine_predict_specific_model(dummy_data, dummy_config, tmp_path):
    train_csv, test_csv = dummy_data

    # 1. Train normally
    engine = TabDynaGluonEngine(config_dict=dummy_config)
    engine.train(train_csv)

    # 2. Get a valid model name that was just trained
    trained_models = engine.predictor.model_names()
    assert len(trained_models) > 0
    specific_model = trained_models[0] # Just pick the first available model

    # 3. Create a special config file with this model name
    special_config_path = tmp_path / 'special_predict_config.yaml'
    special_config_content = f"""
default:
  predict:
    model: "{specific_model}"
"""
    special_config_path.write_text(special_config_content)

    # 4. Load the config similar to how the CLI does
    predict_config = Config.load(str(special_config_path), namespace='predict')

    # 5. Load engine using the config
    loaded_engine = TabDynaGluonEngine.load(
        dummy_config['save_path'],
        config_dict=predict_config.to_dict()
    )

    # 6. Predict (this will internally use the model specified in the config)
    out_csv = str(tmp_path / 'preds_specific.csv')
    preds_df = loaded_engine.predict(test_csv, output_path=out_csv)

    assert os.path.exists(out_csv)
    assert len(preds_df) == 20
    assert loaded_engine.config.get('model') == specific_model


def test_engine_extend(dummy_data, dummy_config, tmp_path):
    train_csv, _ = dummy_data

    # Retain training data for extension (presets: optimize_for_deployment deletes X.pkl)
    dummy_config.get('autogluon_fit_kwargs', {}).pop('presets', None)

    # 1. Initial train
    engine = TabDynaGluonEngine(config_dict=dummy_config)
    engine.train(train_csv)
    original_save_path = Path(dummy_config['save_path'])
    assert (original_save_path / 'run_summary.json').exists()

    # 2. Run extend into a new save_path
    new_save_path = tmp_path / 'AutogluonModels_extended'
    updated_predictor = engine.extend(new_save_path=new_save_path)

    assert updated_predictor is not None
    assert new_save_path.exists()
    assert (new_save_path / 'run.log').exists()
    assert (new_save_path / 'run_summary.json').exists()
    assert (new_save_path / 'leaderboard.parquet').exists()

    # Verify original run directory remains intact
    assert original_save_path.exists()
    assert (original_save_path / 'run.log').exists()


def test_engine_refit(dummy_data, dummy_config, tmp_path):
    train_csv, _ = dummy_data

    # Retain training data for refit (presets: optimize_for_deployment deletes X.pkl)
    dummy_config.get('autogluon_fit_kwargs', {}).pop('presets', None)

    # 1. Initial train
    engine = TabDynaGluonEngine(config_dict=dummy_config)
    engine.train(train_csv)
    original_save_path = Path(dummy_config['save_path'])

    # 2. Run refit into a new save_path
    refit_save_path = tmp_path / 'AutogluonModels_refit'
    refitted_predictor = engine.refit(new_save_path=refit_save_path, model='best')

    assert refitted_predictor is not None
    assert refit_save_path.exists()
    assert (refit_save_path / 'run.log').exists()
    assert not (refit_save_path / 'run_summary.json').exists()
    assert original_save_path.exists()
