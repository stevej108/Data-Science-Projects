import numpy as np
import pymc as pm


print("PyMC test starting...")

with pm.Model() as model:

    x = pm.Normal(
        "x",
        mu=0,
        sigma=1
    )

    y = pm.Normal(
        "y",
        mu=x,
        sigma=1,
        observed=np.array([1.0, 2.0, 3.0])
    )

    idata = pm.sample(
        draws=100,
        tune=100,
        chains=2,
        cores=1,
        progressbar=True,
        random_seed=42
    )


print("SUCCESS")
print(idata)