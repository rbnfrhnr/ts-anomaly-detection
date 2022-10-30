from downstream.rolling_density_singular import SingularRollingDensity


def get_downstream(name, model, **cfg):
    if name.lower() == "srd":
        return SingularRollingDensity(model, **cfg)
