from downstream.rolling_density_singular import SingularRollingDensity
from downstream.discrete_density_dual import DiscreteDualDensity
from downstream.rolling_threshold_singular import SingularRollingThreshold
from downstream.static_threshold_singular import SingularStaticThreshold


def get_downstream(name, model, **cfg):
    if name.lower() == "srd":
        return SingularRollingDensity(model, **cfg)
    if name.lower() == "ddd":
        return DiscreteDualDensity(model, **cfg)
    if name.lower() == "srt":
        return SingularRollingThreshold(model, **cfg)
    if name.lower() == "sst":
        return SingularStaticThreshold(model, **cfg)
