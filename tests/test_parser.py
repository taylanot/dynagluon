from autogluon.common.space import Int, Real

from dynagluon.utils import resolve_config


def test_resolve_config_primitives():
    cfg = {'a': 1, 'b': 'string', 'c': [1, 2, 3]}
    res = resolve_config(cfg)
    assert res == cfg


def test_resolve_config_target():
    cfg = {
        'learning_rate': {
            '_target_': 'autogluon.common.space.Real',
            'lower': 0.01,
            'upper': 0.3,
            'log': True,
        }
    }
    res = resolve_config(cfg)
    assert isinstance(res, dict)
    assert isinstance(res['learning_rate'], Real)
    assert res['learning_rate'].lower == 0.01
    assert res['learning_rate'].upper == 0.3


def test_resolve_config_nested():
    cfg = [
        {'_target_': 'autogluon.common.space.Int', 'lower': 1, 'upper': 10},
        {
            'nested': {
                '_target_': 'autogluon.common.space.Real',
                'lower': 0.1,
                'upper': 1.0,
            }
        },
    ]
    res = resolve_config(cfg)
    assert isinstance(res, list)
    assert isinstance(res[0], Int)
    assert isinstance(res[1], dict)
    assert isinstance(res[1]['nested'], Real)
