"""
Module to transform features by one hot encoding and ordinal encoding

Author: Oluwaseyi E. Ogunnowo
Date: 27th July 2022
"""

import pickle as pkl
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from dependency import functions as func
from dependency import constants as con

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# --- imblearn oversampling and undersampling for class balancing
over = RandomOverSampler(sampling_strategy= 'minority')
under = RandomUnderSampler(sampling_strategy= 'majority')
steps = [('o', over), ('u', under)]
class_balancer = Pipeline(steps=steps)

# --- preprocessing algorithms
one_hot_encoder = OneHotEncoder(handle_unknown = 'ignore')
ordinal_encoder = OrdinalEncoder()
LabelEncoder = LabelEncoder()

logger.info("reading data")
df = pd.read_csv("data_artifacts/features_extracted.csv")

# --- identifying nominal and ordinal columns
logger.info("identifying nominal and ordinal columns")

logger.info("dropping: PhysicalHealth, MentalHealth, BMI, SleepTime from dataset ")
cols_to_drop = ['PhysicalHealth', 'MentalHealth', 'BMI', 'SleepTime']
df.drop(cols_to_drop, axis = 1, inplace = True)

x = df.drop('HeartDisease', axis = 1)
y = df['HeartDisease']

logger.info("encoding nominal features")
one_hot_encoder.fit(x[con.nominal_columns])
ohe_columns = one_hot_encoder.transform(x[con.nominal_columns]).toarray()
ohe_columns = pd.DataFrame(ohe_columns,
                         columns = one_hot_encoder.get_feature_names())

logger.info("encoding categorical features")
ordinal_encoder.fit(x[con.ordinal_columns])
oe_columns = pd.DataFrame(ordinal_encoder.transform(x[con.ordinal_columns]),
                          columns = con.ordinal_columns)

logger.info("encoding target feature")
LabelEncoder.fit(y)
y_transformed = LabelEncoder.transform(y)

x_transformed = pd.concat([ohe_columns, oe_columns], axis = 1)

logger.info("balancing dataset with imblearn")
x_resampled, y_resampled = class_balancer.fit_resample(x_transformed, y_transformed)

x_resampled['HeartDisease'] = y_resampled

logger.info("exporting transformed dataset")
x_resampled.to_csv("data_artifacts/transformed_data.csv", index = False)

logger.info("exporting encoders")
func.export_object(one_hot_encoder, "encoders/one_hot_encoder.pkl")
func.export_object(ordinal_encoder, "encoders/ordinal_encoder.pkl")



