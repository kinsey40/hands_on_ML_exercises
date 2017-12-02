import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
from sklearn import linear_model

warnings.filterwarnings('ignore')

def prepare_country_stats(df1, df2):

    selected_data_df1 = df1[["Country", "Value", "Inequality"]]
    selected_data_df1_rows = selected_data_df1.ix[selected_data_df1["Inequality"] == "Total"]
    selected_data_df2 = df2[["Country", "2015"]]

    selected_data_df1_rows.dropna(how='any', inplace=True)
    selected_data_df2.dropna(how='any', inplace=True)

    selected_data_df1_rows.drop("Inequality", axis=1, inplace=True)

    result = pd.merge(selected_data_df1_rows, selected_data_df2, on='Country', how='inner')
    result.drop_duplicates(inplace=True, keep='first')
    result.columns = ["Country", "Life_Satisfaction", "gdp_per_capita"]

    return result

# Load the data
oecd_bli = pd.read_csv("data/oecd_bli_2015.csv", thousands=",")
gdp_per_capita = pd.read_csv("data/gdp_per_capita.aspx",thousands=",", delimiter='\t',
                            encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["gdp_per_capita"]]
y = np.c_[country_stats["Life_Satisfaction"]]

# Visualize the data
country_stats.plot(kind="scatter", x="gdp_per_capita", y="Life_Satisfaction")
plt.show()

# Select a linear model
model = linear_model.LinearRegression()

# Train the model
model.fit(X, y)

# Make prediction for Cyprus
X_new = [[22587]] # Cyprus' GDP per capita
print(model.predict(X_new))
