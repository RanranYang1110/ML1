import pandas as pd
import os
def load_housing_data(housing_path=r"C:\Users\qianli\datasets\housing"):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
test_set.head()
housing['median_income'].hist()




