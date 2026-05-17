
import pyfair
from pyfair import FairModel, FairSimpleReport
# C:\Python\python.exe -i "$(FULL_CURRENT_PATH)"

# Create using LEF (PERT), PL, (PERT), and SL (PERT)
model1 = pyfair.FairModel(name="Regular Model 1", n_simulations=10_000)
model1.input_data('LEF', low=2, mode=5, high=10)
model1.input_data('PL', low=3_000_000, mode=3_500_000, high=5_000_000)
model1.input_data('SLEF', low=.5, mode=.6, high=.9)
model1.input_data('SLEM', low=1_000_000, mode=1_300_000, high=2_000_000)
model1.calculate_all()

# Create another model using LEF (Normal) and LM (PERT)
model2 = pyfair.FairModel(name="Regular Model 2", n_simulations=10_000)
model2.input_data('Loss Event Frequency', mean=.3, stdev=.1)
model2.input_data('Loss Magnitude', low=2_000_000, mode=3_000_000, high=5_000_000)
model2.calculate_all()

# Create metamodel by combining 1 and 2
mm = pyfair.FairMetaModel(name='My Meta Model!', models=[model1, model2])
mm.calculate_all()

# Create report comparing 2 vs metamodel.
fsr = pyfair.report.simple_report.FairSimpleReport([model1, model2], currency_prefix='EUR')
fsr.to_html('output_MP.html')