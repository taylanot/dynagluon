"""Utility functions for logging and configuration management."""

import logging
import logging.config
from pathlib import Path

from dynaconf import Dynaconf

logger = logging.getLogger(__name__)


def setup_logger(settings: dict[str, object] | None = None) -> None:
    """
    Initialize logging from settings dictionary if present, else fallback.

    Parameters
    ----------
    settings : dict of str to object, optional
        Settings dictionary containing a 'logging' configuration block.
    """
    log_config = settings.get('logging') if settings else None

    if isinstance(log_config, dict):
        logging.config.dictConfig(log_config)
    else:
        # Fallback if no logging section was found in settings
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        )


def load_config(
    path: str | Path = 'settings.yaml',
    namespace: str | None = None,
) -> dict[str, object]:
    """
    Load configuration from a YAML file, returning a standard dictionary.

    Parameters
    ----------
    path : str or Path, default 'settings.yaml'
        File system path to the YAML configuration file.
    namespace : str, optional
        Specific namespace or section to extract (e.g. 'train', 'predict').

    Returns
    -------
    dict of str to object
        Configuration dictionary for the requested namespace, or the full file.
    """
    config = Dynaconf(settings_files=[str(path)], environments=True)

    if namespace:
        val = config.get(namespace, {})
        if hasattr(val, 'to_dict'):
            return val.to_dict()
        if isinstance(val, dict):
            return val
        return {}

    return config.to_dict()
