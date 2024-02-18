import pandas as pd
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_suite import TestSuite

# create data frames from previous and current dataset
prv = pd.read_csv('evidently_ai/sampled_data_prv.csv')
curr = pd.read_csv('evidently_ai/sampled_data_curr.csv')

# Data stability Report
data_stability = TestSuite(tests=[
    DataStabilityTestPreset(),
])
data_stability.run(reference_data=prv, current_data=curr)
data_stability.save_html('data_stability.html')
