"""Utility functions for logging and configuration management."""

from dataclasses import dataclass, field
import importlib
import json
import logging
import logging.config
from pathlib import Path
import sys

from dynaconf import Dynaconf
import yaml

try:
    from rich import print as rprint
except ImportError:
    rprint = None

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Configuration container with serialization and inspection utilities.

    Attributes
    ----------
    data : dict of str to object
        Underlying configuration key-value pairs.
    source_path : Path, optional
        File system path from which this configuration was loaded.
    namespace : str, optional
        Specific section or namespace extracted from the source file.
    env : str, optional
        Active Dynaconf environment used during loading (e.g. 'dev', 'prod').
    """

    data: dict[str, object] = field(default_factory=dict)
    source_path: Path | None = None
    namespace: str | None = None
    env: str | None = None

    @classmethod
    def load(
        cls,
        path: str | Path = 'settings.yaml',
        namespace: str | None = None,
        env: str | None = None,
    ) -> 'Config':
        """
        Load configuration from a YAML file with Dynaconf environment support.

        Parameters
        ----------
        path : str or Path, default 'settings.yaml'
            Path to the YAML configuration file.
        namespace : str, optional
            Specific section or namespace to extract (e.g. 'train', 'predict').
        env : str, optional
            Active environment to load (e.g. 'dev', 'prod', 'development').

        Returns
        -------
        Config
            Loaded Config instance.
        """
        file_path = Path(path)
        kwargs: dict[str, object] = {
            'settings_files': [str(file_path)],
            'environments': True,
        }
        if env:
            kwargs['env'] = env

        dynaconf_obj = Dynaconf(**kwargs)

        if namespace:
            val = dynaconf_obj.get(namespace)
            if val is not None and hasattr(val, 'to_dict'):
                extracted = val.to_dict()
            elif isinstance(val, dict):
                extracted = val
            else:
                extracted = {}

            # If namespace refers to the environment root (e.g. 'default') and no nested
            # subsection was found, fallback to the environment's root settings
            if not extracted and namespace.lower() in (
                'default',
                (env or 'default').lower(),
            ):
                extracted = dynaconf_obj.to_dict()
        else:
            extracted = dynaconf_obj.to_dict()

        return cls(
            data=extracted,
            source_path=file_path,
            namespace=namespace,
            env=env,
        )

    def save(self, destination: str | Path) -> None:
        """
        Save the resolved configuration to a YAML file.

        Parameters
        ----------
        destination : str or Path
            Target file path where configuration will be written.
        """
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open('w', encoding='utf-8') as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)

    def pretty_print(self) -> None:
        """Print formatted configuration to the terminal."""
        if rprint is not None:
            rprint(self.data)
        else:
            sys.stdout.write(
                json.dumps(self.data, indent=2, default=str) + '\n'
            )

    def to_dict(self) -> dict[str, object]:
        """
        Return the raw dictionary.

        Returns
        -------
        dict of str to object
            Configuration dictionary.
        """
        return self.data

    def get(self, key: str, default: object = None) -> object:
        """
        Retrieve an item from the configuration with a default fallback.

        Parameters
        ----------
        key : str
            Configuration key to look up.
        default : object, optional
            Fallback value if key is not found.

        Returns
        -------
        object
            Retrieved value or default.
        """
        if key in self.data:
            return self.data.get(key, default)
        if key.upper() in self.data:
            return self.data.get(key.upper(), default)
        if key.lower() in self.data:
            return self.data.get(key.lower(), default)
        return default

    def __getitem__(self, key: str) -> object:
        """Get an item from the configuration."""
        if key in self.data:
            return self.data[key]
        if key.upper() in self.data:
            return self.data[key.upper()]
        if key.lower() in self.data:
            return self.data[key.lower()]
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the configuration."""
        return (
            key in self.data
            or key.upper() in self.data
            or key.lower() in self.data
        )

    def __bool__(self) -> bool:
        """Check if the configuration is not empty."""
        return bool(self.data)


def setup_logger(
    settings: Config | dict[str, object] | None = None,
    log_file: str | Path | None = None,
) -> None:
    """
    Initialize logging from a Config instance or dictionary if present, else fallback.

    Parameters
    ----------
    settings : Config or dict of str to object, optional
        Settings object containing a 'logging' configuration block.
    log_file : str or Path, optional
        Optional path to a log file where logs should be saved.
    """
    if isinstance(settings, (Config, dict)):
        log_config = settings.get('logging')
    else:
        log_config = None

    if isinstance(log_config, dict):
        logging.config.dictConfig(log_config)
    else:
        # Fallback if no logging section was found in settings
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        )

    # Attach file handler if log_file is provided
    if log_file is not None:
        file_path = Path(log_file).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        root_logger = logging.getLogger()
        already_attached = any(
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() == file_path
            for h in root_logger.handlers
        )
        if not already_attached:
            handler = logging.FileHandler(str(file_path), encoding='utf-8')
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
                )
            )
            root_logger.addHandler(handler)

    # Force AutoGluon to use our logging configuration
    ag_logger = logging.getLogger('autogluon')
    ag_logger.handlers.clear()
    ag_logger.propagate = True


def load_config(
    path: str | Path = 'settings.yaml',
    namespace: str | None = None,
    env: str | None = None,
) -> Config:
    """
    Load configuration from a YAML file, returning a Config instance.

    Parameters
    ----------
    path : str or Path, default 'settings.yaml'
        File system path to the YAML configuration file.
    namespace : str, optional
        Specific namespace or section to extract (e.g. 'train', 'predict').
    env : str, optional
        Active environment to load (e.g. 'dev', 'prod').

    Returns
    -------
    Config
        Loaded Config instance.
    """
    return Config.load(path=path, namespace=namespace, env=env)


def resolve_config(cfg: object) -> object:
    """
    Recursively search a dictionary and dynamically instantiate Python objects.

    If the `_target_` key is present, the dictionary is used to instantiate
    the specified class. Otherwise, it returns the primitive configuration.

    * This function is the corner stone of taking ranges from the configuration
    files. This is relatively slow function as it imports, and you have to make
    sure that you give the full library import e.g. "numpy.random.randn"

    Parameters
    ----------
    cfg : object
        The configuration dictionary, list, or primitive value to resolve.

    Returns
    -------
    object
        The resolved configuration, which may include instantiated objects.

    Raises
    ------
    ValueError
        If an object fails to instantiate.
    """
    if isinstance(cfg, dict):
        # Allow converting dicts that define a dynamic object class
        if '_target_' in cfg:
            # Parse everything else in the dict recursively first (these are kwargs)
            kwargs = {
                k: resolve_config(v) for k, v in cfg.items() if k != '_target_'
            }

            target_str = str(cfg['_target_'])
            try:
                module_name, class_name = target_str.rsplit('.', 1)
                module = importlib.import_module(module_name)
                target_class = getattr(module, class_name)

                logger.debug('Instantiating: %s with %s', target_str, kwargs)
                return target_class(**kwargs)
            except Exception as e:
                raise ValueError(
                    f"Failed to instantiate target '{target_str}': {e}"
                ) from e

        # Standard dictionary recursion
        return {k: resolve_config(v) for k, v in cfg.items()}

    if isinstance(cfg, list):
        # Recurse through lists
        return [resolve_config(item) for item in cfg]

    # Return as-is for primitive types
    return cfg
