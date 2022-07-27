"""
Module contains functions used in this project

Author: Oluwaseyi E. Ogunnowo
Date: 27th July 2022
"""

import pickle as pkl
import pandas as pd

def clean_data(df:pd.DataFrame):
    """
    Function to clean dataframe.
    Fills nulls with median for numerical features and most frequent values for categorical features
    INPUT:
        df: (pd.DataFrame) DataFrame Object
    OUTPUT:
        df: (pd.DataFrame) DataFrame Object
    """
    num_values = df.select_dtypes(include = 'number')
    cat_values = df.select_dtypes(include = 'object')
    for column in num_values:
        df[column] = df[column].fillna(df[column].median())
    for column in cat_values:
        df[column] = df[column].fillna(df[column].value_counts().index[0])
    return df

def load_pickle_obj(path: str):
    """
    Function to unpickle/ deserialize pickled object
    parameters
    ----------
    path (string): path to the pickled object.
    This function can unpickle models and dictionaries
    returns
    ----------
    unpicked object
    """
    with open(path, 'rb') as file:
        return pkl.load(file)
    print(f"object from {path} loaded successfully")
    
def export_object(object_to_pickle: dict, file_name: str):
    """
    Function to pickle/ serialize objects
    Parameters
    ----------
    - data: Object to pickle. This function can pickle dictionary objects as well as ML models
    - file_name: (string) File name to save object as in working directory.
    """
    with open(file_name, 'wb') as _object:
        pkl.dump(object_to_pickle, _object)

def discretize_bmi(bmi_index: int):
    """
    Function to bin BMI value.
    Classifies integer to either underweight, normal, overweight, obese or extremely obese
    INPUT:
        bmi_index: (integer) BMI value
    OUTPUT:
        bmi_level: (string) weight status
    """
    bmi_level = ''
    if bmi_index >= 0 and bmi_index <= 18:
        bmi_level = 'underweight'
    if bmi_index > 18 and bmi_index <= 24:
        bmi_level = 'normal'
    if bmi_index > 24 and bmi_index <= 29:
        bmi_level = 'overweight'
    if bmi_index > 29 and bmi_index <= 39:
        bmi_level = 'obese'
    if bmi_index > 39:
        bmi_level = 'extremely obese'
    return bmi_level

def get_health_concern(index):
    """
    Function to bin number of days in health concern (either mental or physical)
    INPUT:
        index: (integer) Number of days patient is concerned with health
    OUTPUT:
        health_status: (string) concern level
    """
    health_status = ''
    if index >= 0 and index <= 5:
        health_status = 'rarely concerned'
    if index > 5 and index <= 15:
        health_status = 'often concerned'
    if index > 15:
        health_status = 'almost always concerned'
    return health_status

def discretize_num_var(df: pd.DataFrame, column:str  = 'col',
                       steps:int = 6, upper_boundary:int = 24,
                       labels:list = ['list_of_labels']):
    """
    Function to discretize bin times
    INPUT:
        df: (pd.DataFrame) Dataframe object
        column: (string) Name of column to discretize
        steps: (integer) range steps
        upper_boundary: (integer) stop value for range
        labels: (list) labels
    OUTPUT:
        sleep_bins
    """
    sleep_bins = pd.cut(df[column],
                        bins = range(0, 
                                     upper_boundary, 
                                     steps),
                        labels = labels)
    return sleep_bins

def predict(df: pd.DataFrame, model):
    """
    Function to make predictions for streamlit app
    INPUT:
        df: (pd.DataFrame) data for prediction
    OUTPUT:
        result: (integer) probability of heart disease score
    """
    df['sleep_time_bins'] = discretize_num_var(df = df, 
                                               column  = 'SleepTime', steps = 6, 
                                        upper_boundary = 25, 
                                               labels = ['00:00 - 06:00',
                                                         '06:00 - 12:00',
                                                         '12:00 - 18:00',
                                                         '18:00 - 00:00'])
    df['weight_status_by_bmi'] = df['BMI'].apply(lambda x: discretize_bmi(x))
    df['physical_health_status'] = df['PhysicalHealth'].apply(lambda x: get_health_concern(x)) 
    df['mental_health_status'] = df['MentalHealth'].apply(lambda x: get_health_concern(x))
    result = model.predict_proba(df)[:, 1] * 100
    return result
    

