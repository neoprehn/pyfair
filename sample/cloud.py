
import pyfair
from pyfair import FairModel, FairSimpleReport
# C:\Python\python.exe -i "$(FULL_CURRENT_PATH)"

# Create using LEF (PERT), PL, (PERT), and SL (PERT)
model1 = pyfair.FairModel(name="Cloud Business Case", n_simulations=10_000)

model1.input_data('LEF', low=0.45, mode=0.75, high=1) #'Change Event Frequency 0.4 bis jedes Jahr ein Projekt'

# model1.input_data('TEF', low=0.2, mode=0.6, high=3) #'Threat Event Frequency'
# model1.input_data('V', low=0.5, mode=0.6, high= .8) #'Vulnerability'

# model1.input_data('C', low=0.2, mode=0.3, high= .5) #'Contact Frequency'
# model1.input_data('A', low=0.2, mode=0.3, high= .5) #'Probability of Action'

# model1.input_data('TC', low=0.2, mode=0.3, high= .5) #'Threat Capability'
# model1.input_data('CS', low=0.2, mode=0.3, high= .5) #'Control Strength'


# model1.input_data('PL', low=780_000, mode=850_000, high=924_000) #'Primary Win 3,5 MA x 1.600 x 140 Tage / 3,5 MA x 1.200 x 220'
model1.input_data('PL', low=1_008_000, mode=1_200_000, high=1_320_000) #'Primary Win 4,5 MA x 1.600 x 140 Tage / 5 MA x 1.200 x 220'
model1.input_data('SLEF', low=0.2, mode=0.5, high=0.6) #'Secondary Loss Event Frequency'
model1.input_data('SLEM', low=100_000, mode=180_000, high=220_000) #'Secondary Loss Event Magnitude'
model1.calculate_all()

# Create another model using LEF (Normal) and LM (PERT)
model2 = pyfair.FairModel(name="Information Risk Management", n_simulations=10_000)
model2.input_data('LEF', low=0.75, mode=0.8, high=1)
model2.input_data('Loss Magnitude', low=100_000, mode=180_000, high=200_000)
model2.calculate_all()

# Create metamodel by combining 1 and 2
mm = pyfair.FairMetaModel(name='Meine Netsales!', models=[model1, model2])

# Calculate our MetaModel (and contained Models)
mm.calculate_all()

# Export results
#mm.export_results()

# Create report comparing 2 vs metamodel.
fsr = FairSimpleReport([model1, model2, mm], currency_prefix='EUR ')
fsr.to_html('Cloud_Prehn.html')