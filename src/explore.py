import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/california_housing.csv')

sns.set_style('whitegrid')

for column in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')
    plt.show()

for column in df.columns[:-1]:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df, x=column, y='MedHouseVal')
    plt.title(f'{column} vs. MedHouseVal')
    plt.show()

corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

print("Correlation with MedHouseVal:")
print(corr_matrix['MedHouseVal'].sort_values(ascending=False))
