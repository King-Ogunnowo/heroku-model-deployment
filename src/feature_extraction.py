"""
Script to extract features from existing ones

Author: Oluwaseyi E. Ogunnowo
Date: 27th July 2022
"""

import pandas as pd
import logging
from dependency import functions as func

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

logger.info("reading cleaned data")
df = pd.read_csv("data_artifacts/cleaned_data.csv")

logger.info("extracting features: sleep_time_bins, weight_status_by_bmi,\
physical_health_status and mental_health_status")
df['sleep_time_bins'] = func.discretize_num_var(df = df,
                                                column  = 'SleepTime',
                                                steps = 6,
                                                upper_boundary = 25,
                                                labels = ['00:00 - 06:00',
                                                         '06:00 - 12:00',
                                                         '12:00 - 18:00',
                                                         '18:00 - 00:00'])
df['weight_status_by_bmi'] = df['BMI'].apply(lambda x: func.discretize_bmi(x))
df['physical_health_status'] = df['PhysicalHealth'].apply(lambda x: func.get_health_concern(x))
df['mental_health_status'] = df['MentalHealth'].apply(lambda x: func.get_health_concern(x))

logger.info("feature extraction complete, exporting data")
df.to_csv("data_artifacts/features_extracted.csv", index = False)
