import equinox as eqx

def get_param(data, name, GEO_KEYS):
    if name in GEO_KEYS:
        return getattr(data.section, name)
    return getattr(data, name)


def set_param(data, name, new_val, GEO_KEYS):
    if name in GEO_KEYS:
        new_section = eqx.tree_at(lambda s: getattr(s, name), data.section, new_val)
        return eqx.tree_at(lambda d: d.section, data, new_section)
    return eqx.tree_at(lambda d: getattr(d, name), data, new_val)