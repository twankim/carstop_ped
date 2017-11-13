"""
Modifications copyright (C) 2017 UT Austin/Taewan Kim
Factory file for loading config
"""

import config_logitech as cfg_logitech
import config_logitech_curve as cfg_logitech_curve

configs_map = {'config_logitech': cfg_logitech.cfg,
              'config_logitech_curve': cfg_logitech_curve.cfg
              }

def get_config(name):
    """Returns a config options.
    Raises:
        ValueError: If config `name` is not recognized.
    """
    if name not in configs_map:
        raise ValueError('Name of network unknown %s' % name)

    return configs_map[name]
    