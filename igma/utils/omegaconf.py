from omegaconf import OmegaConf


# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
_resolvers = {
    'eq': lambda x, y: x.lower() == y.lower(),
    'contains': lambda x, y: x.lower() in y.lower(),
    'if': lambda pred, a, b: a if pred else b,
    'resolve_default': lambda default, arg: default if arg == '' else arg,
    'resolve_eval': lambda fx, *args: eval(fx)(*args),
}


def register_resolvers(allow_eval=True):
    resolvers = _resolvers.copy()
    if not allow_eval:
        resolvers.pop('resolve_eval')
    for name, resolver in resolvers.items():
        try:
            OmegaConf.register_new_resolver(name, resolver)
        except ValueError:
            continue
