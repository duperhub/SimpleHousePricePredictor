from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

df.to_csv('../data/california_housing.csv', index=False)

def identify_and_handle_outliers(dataframe, column, handle=False):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
    
    if handle:
        dataframe[column] = dataframe[column].clip(lower=lower_bound, upper=upper_bound)
    
    return outliers

for column in df.columns:
    plt.figure(figsize=(10, 4))
    plt.boxplot(df[column])
    plt.title(f'Box plot of {column}')
    plt.show()

for column in df.columns:
    outliers = identify_and_handle_outliers(df, column, handle=True)
    print(f'Outliers in {column}:\n', outliers)

