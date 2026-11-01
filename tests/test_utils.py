import json
import logging
from pathlib import Path
from unittest.mock import patch
import pytest

from dynagluon.utils import Config, setup_logger, load_config, resolve_config

def test_config_basic(tmp_path):
    cfg = Config(data={'a': 1, 'b': 2})
    assert cfg['a'] == 1
    assert 'b' in cfg
    assert bool(cfg) is True
    
    cfg_empty = Config()
    assert bool(cfg_empty) is False
    
    dest = tmp_path / 'test_save.yaml'
    cfg.save(dest)
    assert dest.exists()

def test_config_pretty_print(capsys):
    cfg = Config(data={'a': 1})
    with patch('dynagluon.utils.rprint', None):
        cfg.pretty_print()
        out, _ = capsys.readouterr()
        assert '{\n  "a": 1\n}\n' in out

def test_config_pretty_print_rich():
    cfg = Config(data={'a': 1})
    with patch('dynagluon.utils.rprint') as mock_rprint:
        cfg.pretty_print()
        mock_rprint.assert_called_once_with({'a': 1})

def test_setup_logger(tmp_path):
    # test default fallback
    setup_logger()

    # test with dict config
    log_cfg = {
        'logging': {
            'version': 1,
            'loggers': {
                'test_logger': {'level': 'DEBUG'}
            }
        }
    }
    setup_logger(Config(data=log_cfg))

    # test with log_file parameter
    log_file = tmp_path / 'custom_run.log'
    setup_logger(log_file=log_file)
    logging.getLogger().info('Log file test message')
    assert log_file.exists()
    assert 'Log file test message' in log_file.read_text(encoding='utf-8')

def test_load_config_helper():
    yaml_path = Path(__file__).parent / 'test_configs' / 'test_utils_config.yaml'
    cfg = load_config(str(yaml_path), namespace='default')
    assert cfg['test_key'] == 42

def test_resolve_config_error():
    cfg = {
        '_target_': 'nonexistent.module.ClassName',
        'kwarg': 1
    }
    with pytest.raises(ValueError, match="Failed to instantiate target"):
        resolve_config(cfg)

