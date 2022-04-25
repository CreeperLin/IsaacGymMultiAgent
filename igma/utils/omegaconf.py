from omegaconf import OmegaConf


def register_resolvers(allow_eval=True):
    # Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
    OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
    # allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
    # num_ensv
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)
    if allow_eval:
        OmegaConf.register_new_resolver('resolve_eval', lambda fx, *args: eval(fx)(*args))
