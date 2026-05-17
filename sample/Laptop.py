import pyfair
from pyfair import FairModel, FairSimpleReport

# Create a model
model1 = FairModel(name='Laptop Diebstahl', n_simulations=10_000)
# model1.input_data('Loss Event Frequency', mean=.3, stdev=.1)
# model1.input_data('CF', low=0, mode=1, high=5) #'Contact Frequency'
# model1.input_data('PoA', low=0, mode=1, high=5) #'Probability of Action'
model1.input_data('TEF', low=0, mode=1, high=5) #'Threat Event Frequency'

# model1.input_data('V', low=0.5, mode=0.6, high= .8) #'Vulnerability'
model1.input_data('TC', low=0.2, mode=0.3, high= .5) #'Threat Capability'
model1.input_data('CS', low=0.2, mode=0.3, high= .5) #'Control Strength'

model1.input_data('PL', low=3_700, mode=76_750, high=252_500) #'Primary Loss'
model1.input_data('SLEF', low=.05, mode=.06, high=.1) #'Secondary Loss Event Frequency'
model1.input_data('SLEM', low=100_000, mode=250_000, high=1_800_000) #'Secondary Loss Event Magnitude'

model1.calculate_all()

# Create a model 2
model2 = FairModel(name='Laptop verschlüsselt', n_simulations=10_000)
model2.input_data('TEF', low=0, mode=1, high=5) #'Threat Event Frequency'
# model2.input_data('V', low=0.5, mode=0.6, high= .8) #'Vulnerability'
model2.input_data('TC', low=0.3, mode=0.5, high= .7) #'Threat Capability'
model2.input_data('CS', low=0.5, mode=0.7, high= .8) #'Control Strength'

model2.input_data('PL', low=3_700, mode=76_750, high=252_500) #'Primary Loss'
model2.input_data('SLEF', low=.05, mode=.06, high=.1) #'Secondary Loss Event Frequency'
model2.input_data('SLEM', low=100_000, mode=250_000, high=1_800_000) #'Secondary Loss Event Magnitude'

model2.calculate_all()

# Create metamodel by combining 1 and 2
mm = pyfair.FairMetaModel(name='Vergleich', models=[model1, model2])

# Calculate our MetaModel (and contained Models)
mm.calculate_all()

# Create a report and write it to an output.
#fsr = FairSimpleReport([model1])
fsr = FairSimpleReport([model1, model2, mm], currency_prefix='EUR ')
fsr.to_html('laptop.html')
