import pandas as pd
from evidently.test_preset import DataQualityTestPreset
from evidently.test_suite import TestSuite

# create data frames from previous and current dataset
prv = pd.read_csv('evidently_ai/sampled_data_prv.csv')
curr = pd.read_csv('evidently_ai/sampled_data_curr.csv')

# Data Quality Report
data_quality = TestSuite(tests=[
    DataQualityTestPreset(),
])

data_quality.run(reference_data=prv, current_data=curr)
data_quality.save_html('data_quality.html')
