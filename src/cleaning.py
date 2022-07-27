"""
Module to clean training data

Author: Oluwaseyi E. Ogunnowo
Date: 27th July 2022
"""

import pandas as pd
import logging
from dependency import functions as func

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

logger.info("reading data")
df = pd.read_csv("sampled_data.csv")

logger.info("cleaning training data")
df = func.clean_data(df)

logger.info("exporting cleaned data")
df.to_csv("./data_artifacts/cleaned_data.csv", index = False)
