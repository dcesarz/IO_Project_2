import os

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(THIS_FOLDER, 'HealthInsuranceCrossSellPrediction.csv')


def load():
    df = pd.read_csv(file)
    df = df.drop(labels=['id'], axis=1)
    return df


#   utility function for record checking
def check(df):
    x = 0
    for i in df['Driving_License']:  # there's a lot of 1s in this column, let's check if there are any 0s.
        if i == 0:
            print('0 FOUND, COLUMN ISN\'T OBSOLUTE')
            break
    print(np.where(pd.isnull(df)))  # NO MISSING RECORDS
    # CHECK RANGE
    z = df['Vintage'].unique()
    print(np.sort(z))

    z = df['Annual_Premium'].unique()
    print(np.sort(z))
    for column in df:
        print(column)
        print(df[column].isna().values.any())


def normalize(df):
    df_new = df.copy()
    cols = df_new.columns
    df_new['Vehicle_Damage'] = df_new['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
    df_new['Vehicle_Age'] = df_new['Vehicle_Age'].map({'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0})
    df_new['Gender'] = df_new['Gender'].map({'Female': 1, 'Male': 0})
    all_inputs = df_new[
        ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
         'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']].values
    all_classes = df_new['Response'].values
    df_new = df_new.apply(LabelEncoder().fit_transform)
    vals = df_new.values
    vals = MinMaxScaler().fit_transform(vals)
    new_df = DataFrame(vals)
    new_df = new_df.apply(round, args=(2,))
    new_df.columns = cols

    new_df.to_csv('classification.csv')

    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(
        all_inputs, all_classes, train_size=0.7)

    return train_inputs, test_inputs, train_classes, test_classes


def classification_prep(df):
    df_new = df.copy()
    train_inputs, test_inputs, train_classes, test_classes = normalize(df_new)
    return train_inputs, test_inputs, train_classes, test_classes


def association_prep(df):
    new_df = df.copy()

    age_labels = ['< 30', '30 - 60 ', '60 +']
    premium_labels = ['< 20 000', '20 001 - 40 000', '40 001 - 80 000', '80 001 - 100 000', '100 001 - 250 000',
                      '250 000 +']
    vintage_labels = ['< 100', '101 - 200', '200+']

    age_bins = [0, 31, 61, 200]
    premium_bins = [0, 20001, 40001, 80001, 100001, 250001, 99999999]
    vintage_bins = [0, 101, 201, 99999]

    new_df['Age'] = pd.cut(
        new_df['Age'], age_bins, labels=age_labels)
    new_df['Annual_Premium'] = pd.cut(
        new_df['Annual_Premium'], premium_bins, labels=premium_labels)
    new_df['Vintage'] = pd.cut(
        new_df['Vintage'], vintage_bins, labels=vintage_labels)

    rc = df['Region_Code'].unique()
    psc = df['Policy_Sales_Channel'].unique()

    rc.sort()
    psc.sort()

    rc_count = len(rc)
    psc_count = len(psc)

    rc_labels = []
    psc_labels = []
    for i in range(rc_count - 1):
        s = "region_{}".format(i + 1)
        rc_labels.append(s)

    for i in range(psc_count - 1):
        s = "channel_{}".format(i + 1)
        psc_labels.append(s)
    new_df['Region_Code'] = pd.cut(new_df['Region_Code'], rc, labels=rc_labels)
    new_df['Policy_Sales_Channel'] = pd.cut(new_df['Policy_Sales_Channel'], psc, labels=psc_labels)

    new_df = pd.get_dummies(new_df, prefix_sep='_', drop_first=True)
    print(new_df.head())
    new_df.to_csv("association.csv")
    return new_df
