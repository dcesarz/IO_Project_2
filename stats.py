import pandas as pd


def get_stats(df):
    statistics = {}
    values = {}
    new_df = df.copy()
    mean = []
    max = []
    min = []

    numeric = ['Age', 'Annual_Premium', 'Vintage']

    for column in new_df:
        if column not in numeric:
            pass
            vals = new_df[column].value_counts(normalize=True)
            values[column] = vals

    for column in numeric:
        mean.append(new_df[column].mean())
        max.append(new_df[column].max())
        min.append(new_df[column].min())

    statistics['mean'] = mean
    statistics['max'] = max
    statistics['min'] = min

    statistics = pd.DataFrame.from_dict(statistics)

    statistics.index = numeric

    age_labels = ['< 30', '30 - 60 ', '60 +']
    premium_labels = ['< 10 000', '10 001 - 20 000', '20 001 - 40 000', '40 001 - 80 000', '80 001 - 100 000',
                      '100 000 +']
    vintage_labels = ['< 100', '101 - 200', '200+']

    age_bins = [0, 31, 61, 200]
    premium_bins = [0, 10001, 20001, 40001, 80001, 100001, 99999999]
    vintage_bins = [0, 101, 201, 99999]

    new_df['Age'] = pd.cut(
        new_df['Age'], age_bins, labels=age_labels)
    new_df['Annual_Premium'] = pd.cut(
        new_df['Annual_Premium'], premium_bins, labels=premium_labels)
    new_df['Vintage'] = pd.cut(
        new_df['Vintage'], vintage_bins, labels=vintage_labels)

    for column in numeric:
        vals = new_df[column].value_counts(normalize=True)
        values[column] = vals

    return statistics, values
