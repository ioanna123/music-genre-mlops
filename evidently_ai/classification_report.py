import pandas as pd
from evidently.metric_preset import ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report

# create data frames from previous and current dataset
prv = pd.read_csv('evidently_ai/sampled_data_prv.csv')
curr = pd.read_csv('evidently_ai/sampled_data_curr.csv')

# create column mapping
column_mapping = ColumnMapping()

column_mapping.target = 'label'
column_mapping.prediction = 'prediction'
column_mapping.numerical_features = ["length", "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var",
                                     "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean",
                                     "spectral_bandwidth_var",
                                     "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean",
                                     "zero_crossing_rate_var", "harmony_mean",
                                     "harmony_var", "perceptr_mean", "perceptr_var", "tempo", "mfcc1_mean",
                                     "mfcc1_var", "mfcc2_mean",
                                     "mfcc2_var", "mfcc3_mean", "mfcc3_var", "mfcc4_mean", "mfcc4_var",
                                     "mfcc5_mean", "mfcc5_var",
                                     "mfcc6_mean", "mfcc6_var", "mfcc7_mean", "mfcc7_var", "mfcc8_mean",
                                     "mfcc8_var", "mfcc9_mean",
                                     "mfcc9_var", "mfcc10_mean", "mfcc10_var", "mfcc11_mean", "mfcc11_var",
                                     "mfcc12_mean", "mfcc12_var",
                                     "mfcc13_mean", "mfcc13_var", "mfcc14_mean", "mfcc14_var", "mfcc15_mean",
                                     "mfcc15_var", "mfcc16_mean",
                                     "mfcc16_var", "mfcc17_mean", "mfcc17_var", "mfcc18_mean", "mfcc18_var",
                                     "mfcc19_mean", "mfcc19_var",
                                     "mfcc20_mean", "mfcc20_var"]
column_mapping.categorical_features = ["filename", "label", "prediction"]

# create classification report
classification_performance_report = Report(metrics=[
    ClassificationPreset(),
])

classification_performance_report.run(reference_data=prv, current_data=curr,
                                      column_mapping=column_mapping)
classification_performance_report.save_html('classification_report.html')
print(classification_performance_report)
