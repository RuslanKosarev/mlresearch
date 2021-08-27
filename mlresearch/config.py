# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

import sys
from typing import Union, Tuple
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf, DictConfig

import torch

import random
import numpy as np

PathType = Union[Path, str]

# directory for default configs
default_config_dir = Path(__file__).parents[0].joinpath('configs')
default_config = default_config_dir / 'config.yaml'

user_config_dir = Path(__file__).parents[1] / 'configs'
user_config = user_config_dir / 'config.yaml'


def subdir() -> str:
    return datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')


def set_seed(seed: int = None):
    """
    set seed for random number generators

    :param seed:
    :return:
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(0)


class Config:
    """Object representing YAML settings as a dict-like object with values as fields
    """

    def __init__(self, dct: dict = None):
        """Update config from dict
        :param dct: input object
        """
        if dct is None:
            dct = dict()

        for key, item in dct.items():
            if isinstance(item, dict):
                setattr(self, key, Config(item))
            else:
                setattr(self, key, item)

    def __repr__(self):
        shift = 3 * ' '

        def get_str(obj, ident=''):
            s = ''
            for key, item in obj.items():
                if isinstance(item, Config):
                    s += f'{ident}{key}: \n{get_str(item, ident=ident + shift)}'
                else:
                    s += f'{ident}{key}: {str(item)}\n'
            return s

        return get_str(self)

    def __getattr__(self, name):
        return self.__dict__.get(name, Config())

    def __bool__(self):
        return bool(self.__dict__)

    @property
    def as_dict(self):
        def as_dict(obj):
            s = {}
            for key, item in obj.items():
                if isinstance(item, Config):
                    item = as_dict(item)
                s[key] = item
            return s

        return as_dict(self)

    def items(self):
        return self.__dict__.items()


class LoadConfigError(Exception):
    pass


def config_paths(app_file_name, custom_config_file=None):
    config_name = Path(app_file_name).stem + '.yaml'

    paths = [
        default_config,
        default_config_dir.joinpath(config_name),
        user_config,
        user_config_dir.joinpath(config_name)
    ]

    if custom_config_file is not None:
        paths.append(custom_config_file)

    return tuple(paths)


def application_name():
    return Path(sys.argv[0]).stem


def load_config(dct: dict = None, file: PathType = None):
    """Load configuration from the set of config files
    :param dct:  Optional dictionary
    :param file: Optional path to the custom config file
    :return: The validated config in Config model instance
    """

    paths = config_paths(application_name(), file)

    cfg = OmegaConf.create()

    for config_path in paths:
        if not config_path.is_file():
            continue

        try:
            new_cfg = OmegaConf.load(config_path)
            cfg = OmegaConf.merge(cfg, new_cfg)
        except Exception as err:
            raise LoadConfigError(f"Cannot load configuration from '{config_path}'\n{err}")

    cfg = OmegaConf.merge(cfg, dct)

    if len(cfg.keys()) == 0:
        raise LoadConfigError("The configuration has not been loaded.")

    options = OmegaConf.to_container(cfg)
    options = Config(options)

    set_seed(seed=options.seed)

    return options
