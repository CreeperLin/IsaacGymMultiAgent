import os
import igma
import inspect


igma_root_dir = os.path.dirname(inspect.getfile(igma))


def get_cfg_dir():
    return os.path.join(igma_root_dir, 'cfg')


def get_assets_dir():
    return os.path.join(igma_root_dir, 'assets')
