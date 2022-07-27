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

df = pd.read_csv("data_artifacts/transformed_data.csv")

# x = df.drop('HeartDisease', axis = 1)
# y = df['HeartDisease']

logger.info("splitting data into train and test sets")
train_df = df.sample(frac = 0.8)
df.drop(train_df.index, axis = 0, inplace = True)
test_df = df.sample(frac = 0.2)

print(train_df['HeartDisease'].value_counts())
print(test_df['HeartDisease'].value_counts())



# x_train, x_test, y_train, y_test = train_test_split(x, y,
#                                                    test_size = 0.2,
#                                                    stratify = y)

# x_train['HeartDisease'] = y_train
# x_test['HeartDisease'] = y_test

# logger.info("exporting training set")
# x_train.to_csv("data_artifacts/train_set.csv", index = False)

# logger.info("exporting testing set")
# x_test.to_csv("data_artifacts/test_set.csv", index = False)
