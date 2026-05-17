import traceback

import pyfair
from pyfair import FairModel, FairSimpleReport
import pyfair.report.tree_graph as tg

# ------------------------------------------------------------
# Model
# New API with explicit shape/range parameters
# -> TEF should be calculated from CF and PoA
# -> Vulnerability should be calculated from TC and CS
# ------------------------------------------------------------
model2 = FairModel(name='Laptop', n_simulations=10_000)

model2.input_data(
    'CF',
    distribution='poisson',
    params={'lambda': 2.5, 'range': 0.25}
)

model2.input_data(
    'PoA',
    distribution='beta',
    params={'mean': 0.40, 'k': 40}
)

model2.input_data(
    'TC',
    distribution='beta',
    params={'mean': 0.50, 'k': 25}
)

model2.input_data(
    'CS',
    distribution='beta',
    params={'mean': 0.70, 'k': 25}
)

model2.input_data(
    'PL',
    distribution='lognormal',
    params={'mean': 76_750, 'sigma': 0.9}
)

model2.input_data(
    'SLEF',
    distribution='beta',
    params={'mean': 0.06, 'k': 30}
)

model2.input_data(
    'SLEM',
    distribution='pert',
    params={'low': 100_000, 'mode': 250_000, 'high': 1_000_000, 'gamma': 3}
)

model2.calculate_all()

# ------------------------------------------------------------
# Risk Tolerance
# ------------------------------------------------------------
# risk_tolerance = {
    # "type": "constant",
    # "value": 5000
# }

# risk_tolerance = {
    # "type": "curve",
    # "points": [
        # {"loss": 1_000, "level": 1.00},
        # {"loss": 1_800, "level": 0.80},
        # {"loss": 5_000, "level": 0.50},
        # {"loss": 12_000, "level": 0.20},
        # {"loss": 20_000, "level": 0.10},
    # ]
# }

# risk_tolerance = {
    # "type": "distribution",
    # "distribution": "pert",
    # "params": {
        # "low": 1_000,
        # "mode": 2_000,
        # "high": 10_000,
        # "gamma": 4.0
    # },
    # "samples": 100000
# }

risk_tolerance = {
    "type": "distribution",
    "distribution": "lognormal",
    "params": {
        "mean": 5_000,
        "sigma": 0.5
    },
    "samples": 20000
}

# ------------------------------------------------------------
# Report generation
# nur html fsr.to_html("report.html", export_csv=True/False)
# HTML + CSV fsr.to_html("report.html")
# nur CSV fsr.to_html("report.html", export_csv=True/False)
# Direkt am Modell fsr.to_csv(output_dir="exports")
# ------------------------------------------------------------
try:
    fsr = FairSimpleReport(
        [model2],
        currency_prefix='EUR ',
        risk_tolerance=risk_tolerance
    )
    fsr.to_html("results/Laptop_neu.html", export_csv=False)
    print("\nReport created successfully: laptop_neu.html")
except Exception as e:
    print("\nReport generation failed.")
    print("Error:", e)
    traceback.print_exc()

