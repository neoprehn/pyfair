from pyfair import FairModel, FairSimpleReport

# Create a model
model1 = FairModel(name='APT', n_simulations=10_000)
# model1.input_data('Loss Event Frequency', low=0.2, mode=0.3, high=0.5)
model1.input_data('TEF', low=0.4, mode=0.5, high=0.8) #'Threat Event Frequency'
# model1.input_data('V', low=0.5, mode=0.6, high= .8) #'Vulnerability'
model1.input_data('TC', low=0.6, mode=0.65, high= .9) #'Threat Capability'
model1.input_data('CS', low=0.75, mode=0.92, high= .99) #'Control Strength'

model1.input_data('PL', low=14_000_000, mode=35_250_000, high=50_000_000) #'Primary Loss'
model1.input_data('SLEF', low=0, mode=0.06, high=0.06) #'Secondary Loss Event Frequency'
model1.input_data('SLEM', low=0, mode=1, high=1) #'Secondary Loss Event Magnitude'
model1.calculate_all()

# Create a report and write it to an output.
fsr = FairSimpleReport([model1])
fsr.to_html('pbb_output.html')