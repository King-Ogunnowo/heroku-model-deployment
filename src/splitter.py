"""
Module to split cleaned data into train and test sets

Author: Oluwaseyi E. Ogunnowo
Date: 27th July 2022
"""

import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

logger.info("reading data")
df = pd.read_csv("data_artifacts/transformed_data.csv")

logger.info("splitting data into train and test sets")
train_df = df.sample(frac = 0.8)
df.drop(train_df.index, axis = 0, inplace = True)
test_df = df.sample(frac = 0.2)

logger.info("exporting training set")
train_df.to_csv("data_artifacts/train_set.csv", index = False)

logger.info("exporting testing set")
test_df.to_csv("data_artifacts/test_set.csv", index = False)
