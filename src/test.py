"""
Module to validate model

Author: Oluwaseyi E. Ogunnowo
Date: 27th July 2022
"""

import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from dependency import functions as func

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

logger.info("reading data")
df = pd.read_csv("data_artifacts/test_set.csv")

logger.info("loading model")
model = func.load_pickle_obj("model/model.pkl")

x = df.drop("HeartDisease", axis = 1)
y = df['HeartDisease']

logger.info("predicting binary outputs")
y_pred = model.predict(x)

logger.info("predicting probabilities")
probability = model.predict_proba(x)[:, 1]
  
logger.info("generating classification report and storing in results folder")
plt.figure(figsize=(6, 5))
plt.text(0.01, 1.05, str("Classification Report"),{'fontsize': 10}, fontproperties='monospace')
plt.text(0.01, 0.05, str(classification_report(y, y_pred)), {'fontsize': 10}, fontproperties='monospace')
plt.axis('off')
plt.savefig('results/classification_report.png')

logger.info("plotting feature importance")
func.feature_imp(x, model = model, num_of_feats = 15)

logger.info("plotting calibration curve")
func.plot_calibration_curve(y, probability, bins = 10, model_name = 'Random Forest Classifier')







