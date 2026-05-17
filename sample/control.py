import pyfair
from pyfair import FairModel, FairSimpleReport

# Create a model
model = FairModel(name='Control', n_simulations=10_000)
# CF
# Annahme: aktuelles Modell kann Poisson mit lambda/range,
# aber kein natives "lambda ~ PERT".
model.input_data(
    'CF',
    distribution='pert',
    params={
        'low': 120,
        'mode': 180,
        'high': 250
    }
)

# PoA
model.input_data(
    'PoA',
    distribution='beta',
    params={
        'mean': 0.02,
        'k': 15
    }
)

# TC
model.input_data(
    'TC',
    distribution='pert',
    params={
        'low': 0.55,
        'mode': 0.75,
        'high': 0.90
    }
)

# RS -> als CS gemappt
model.input_data(
    'CS',
    distribution='pert',
    params={
        'low': 0.35,
        'mode': 0.55,
        'high': 0.75
    }
)

# PLM
# Aus P50=400k und P90=1.4M hergeleitet:
# sigma ≈ 0.9775
# arithmetic mean ≈ 645001.73
model.input_data(
    'PL',
    distribution='lognormal',
    params={
        'mean': 645001.73,
        'sigma': 0.9775
    }
)

# SLEF
model.input_data(
    'SLEF',
    distribution='beta',
    params={
        'mean': 0.35,
        'k': 10
    }
)

# SLM
# Aus P50=150k und P90=600k hergeleitet:
# sigma ≈ 1.0817
# arithmetic mean ≈ 269267.83
model.input_data(
    'SLEM',
    distribution='lognormal',
    params={
        'mean': 269267.83,
        'sigma': 1.0817
    }
)

model.calculate_all()

risk_tolerance = {
    "type": "distribution",
    "distribution": "lognormal",
    "params": {
        "mean": 10_000,
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
        [model],
        currency_prefix='EUR ',
        risk_tolerance=risk_tolerance
    )
    fsr.to_html("results/control.html", export_csv=True)
    print("\nReport created successfully")
except Exception as e:
    print("\nReport generation failed.")
    print("Error:", e)
    traceback.print_exc()