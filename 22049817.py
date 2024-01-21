import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import FuncFormatter

df_countries=pd.read_csv('C:\\Users\\DELL\\Downloads\\3Cluster\\API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_6299951.csv',skiprows=4)

def plot_country_data(country_name, indicator_name, df_countries):
    # Extract years and indicator data for the specified country and indicator
    country_data = df_countries[(df_countries['Country Name'] == country_name) & (df_countries['Indicator Name'] == indicator_name)]
    years = country_data.columns[4:]  # Assuming the years start from the 5th column
    indicator_data = country_data.iloc[:, 4:].values.flatten()

    # Convert years to numeric values
    years_numeric = pd.to_numeric(years, errors='coerce')
    indicator_data = pd.to_numeric(indicator_data, errors='coerce')

    # Remove rows with NaN or inf values
    valid_data_mask = np.isfinite(years_numeric) & np.isfinite(indicator_data)
    years_numeric = years_numeric[valid_data_mask]
    indicator_data = indicator_data[valid_data_mask]

    # Define the model function
    def indicator_model(year, a, b, c):
        return a * np.exp(b * (year - 1990)) + c

    # Curve fitting with increased maxfev
    params, covariance = curve_fit(indicator_model, years_numeric, indicator_data, p0=[1, -0.1, 90], maxfev=10000)

    # Optimal parameters
    a_opt, b_opt, c_opt = params

    # Generate model predictions for the year 2040
    future_years = np.arange(min(years_numeric), 2041, 1)
    indicator_future = indicator_model(future_years, a_opt, b_opt, c_opt)

    # Create a combined bar and line plot
    plt.figure(figsize=(14, 8))
    plt.bar(years_numeric, indicator_data, color='skyblue', edgecolor='darkblue', label='Actual Data')
    plt.plot(future_years, indicator_future, color='limegreen', linestyle='--', label='Prediction for 2040')

    # Add labels and title
    plt.title(f'{indicator_name} Trends and Prediction in {country_name}', fontsize=18, color='navy')
    plt.xlabel('Year', fontsize=14, color='navy')
    plt.ylabel(indicator_name, fontsize=14, color='navy')

    # Beautify the plot
    sns.set(style="whitegrid")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    # Create a pie chart for trends and predictions
    plt.figure(figsize=(8, 8))
    labels = ['Actual Data', 'Prediction for 2040']
    sizes = [len(years_numeric), len(future_years)]
    colors = ['skyblue', 'limegreen']
    explode = (0.1, 0)  # explode 1st slice
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title(f'Distribution of Trends and Prediction in {country_name}', fontsize=16, color='navy')

    # Show the plots
    plt.show()

# Example usage:
indicator_name = 'Access to electricity (% of population)'
countries = ['Kenya', 'Moldova', 'Mexico']

for country in countries:
    plot_country_data(country, indicator_name, df_countries)

# CLUSTER ANALYSIS OF ELECTRICITY DATA
# Extract data for the years 1999 and 2020
years = ['1999', '2020']

# Extract relevant data
inflation_data = df_countries[['Country Name'] + years].dropna()

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(inflation_data.drop('Country Name', axis=1))

# Perform KMeans clustering
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(normalized_data)

# Add cluster labels to the DataFrame
inflation_data['Cluster'] = labels

# Visualize the clusters in a more stylish way
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")

# Melt the DataFrame for better visualization
melted_data = inflation_data.melt(id_vars=['Country Name', 'Cluster'], var_name='Year', value_name='Access to electricity (%)')

# Plot using a categorical plot
sns.catplot(x='Year', y='Access to electricity (%)', hue='Cluster', data=melted_data, kind='swarm', palette='Set1', height=6, aspect=2)
plt.title('Cluster Analysis of Access to Electricity Data', fontsize=16, color='navy')
plt.xlabel('Year', color='darkblue', fontsize=14)
plt.ylabel('Access to electricity (%)', color='darkblue', fontsize=14)
plt.show()