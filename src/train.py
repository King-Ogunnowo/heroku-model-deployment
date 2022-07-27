"""
Module to train ML algorithm

Author: Oluwaseyi E. Ogunnowo
Date: 27th July 2022
"""

import pandas as pd
import pickle as pkl
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import logging
from dependency import functions as func

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

rfc_clf = RandomForestClassifier(random_state = 42)

logger.info("reading data")
df = pd.read_csv("data_artifacts/train_set.csv")

x = df.drop("HeartDisease", axis = 1)
y = df['HeartDisease']

logger.info("training classifier")
rfc_clf.fit(x, y)

logger.info("exporting model")
func.export_object(rfc_clf, "model/model.pkl")



