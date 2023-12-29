import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/california_housing.csv')
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)