from downstream.rolling_density_singular import SingularRollingDensity
from downstream.discrete_density_dual import DiscreteDualDensity


def get_downstream(name, model, **cfg):
    if name.lower() == "srd":
        return SingularRollingDensity(model, **cfg)
    if name.lower() == "ddd":
        return DiscreteDualDensity(model, **cfg)
