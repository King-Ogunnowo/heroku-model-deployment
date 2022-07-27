"""
Script to compile model and encoders into model pipeline

Author: Oluwaseyi E. Ogunnowo
Date: 27th July 2022
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import logging
from dependency import functions as func
from dependency import constants as con

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

logger.info("reading data")
df = pd.read_csv("data_artifacts/features_extracted.csv")

logger.info("loading model and encoders")
model = func.load_pickle_obj("model/model.pkl")
one_hot_encoder = func.load_pickle_obj("encoders/one_hot_encoder.pkl")
ordinal_encoder = func.load_pickle_obj("encoders/ordinal_encoder.pkl")

train_df = df.sample(frac = 0.8)

x = train_df.drop("HeartDisease", axis = 1)
y = train_df['HeartDisease']

logger.info("creating model pipeline object")
transformer = ColumnTransformer([('onehotencoding', one_hot_encoder, con.nominal_columns),
                                ('ordinal_encoding', ordinal_encoder, con.ordinal_columns)])
model_pipeline = Pipeline([('transform_variables', transformer),
                        ('make_predictions', model)])

logger.info("fitting model pipeline object")
model_pipeline.fit(x, y)

logger.info("exporting model pipeline object")
func.export_object(model_pipeline, "model/model_pipeline.pkl")