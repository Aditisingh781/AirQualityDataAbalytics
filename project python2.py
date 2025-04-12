
"""
Created on Fri Apr 11 22:01:11 2025

@author: Aditi singh
"""

# Import all required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Step 1: Load the Data
print("Loading the data...")
# Load data from Excel file
df = pd.read_excel(r"c:\Users\Aditi singh\OneDrive\Desktop\PYTHON CA2.xlsx")  # Replace with your file path
print("Data loaded successfully!")
print("\nFirst 5 rows of the dataset:\n", df.head())

# Step 2: Data Cleaning
print("\n--- Data Cleaning ---")

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Fill missing numeric values with 0 (e.g., pollutant metrics)
numeric_columns = ['pollutant_min', 'pollutant_max', 'pollutant_avg', 'latitude', 'longitude']
df[numeric_columns] = df[numeric_columns].fillna(0)

# Convert 'last_update' to datetime (assuming Excel serial date format as per the data)
if pd.api.types.is_numeric_dtype(df['last_update']):
    df['last_update'] = pd.to_datetime(df['last_update'], unit='D', origin='1899-12-30')
else:
    print("Error: 'last_update' column contains non-numeric data. Attempting to parse as string.")
    df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')

# Check data types
print("\nData types:\n", df.dtypes)

# Remove duplicates if any
df = df.drop_duplicates()
print("\nNumber of duplicates removed:", len(df) - len(df.drop_duplicates()))

# Step 3: Exploratory Data Analysis (EDA)
print("\n--- Exploratory Data Analysis ---")

# Summary statistics for numeric columns
print("\nSummary Statistics:\n", df[numeric_columns].describe())

# Total pollutant average by state
print("\nTotal Pollutant Average by State:\n", df.groupby('state')['pollutant_avg'].sum().sort_values(ascending=False))

# Total pollutant average by city
print("\nTop 10 Cities by Pollutant Average:\n", df.groupby('city')['pollutant_avg'].sum().sort_values(ascending=False).head(10))

# Pollutant measurements per pollutant type
df['pollutant_id'] = df['pollutant_id'].fillna('Unknown')  # Handle missing pollutant IDs
print("\nPollutant Measurements by Type:\n", df['pollutant_id'].value_counts())

# Top 10 stations by pollutant average
print("\nTop 10 Stations by Pollutant Average:\n", df[['station', 'pollutant_avg']].sort_values(by='pollutant_avg', ascending=False).head(10))

# Step 4: Visualizations
print("\n--- Visualizations ---")
sns.set_style("whitegrid")

# Bar chart for top 10 stations by pollutant average
if 'pollutant_avg' in df.columns and 'station' in df.columns:
    plt.figure(figsize=(12, 6))
    top_10_stations = df.nlargest(10, 'pollutant_avg')
    sns.barplot(x='pollutant_avg', y='station', data=top_10_stations, palette='viridis')
    plt.title('Top 10 Stations by Pollutant Average')
    plt.xlabel('Pollutant Average')
    plt.ylabel('Station')
    plt.show()
else:
    print("Error: Required columns 'pollutant_avg' or 'station' are missing in the DataFrame.")

# Scatter plot for latitude vs pollutant average by state
plt.figure(figsize=(12, 6))
sns.scatterplot(x='latitude', y='pollutant_avg', hue='state', size='pollutant_avg', data=df, sizes=(20, 200))
plt.title('Latitude vs Pollutant Average by State')
plt.xlabel('Latitude')
plt.ylabel('Pollutant Average')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Pie chart for pollutant type distribution
plt.figure(figsize=(8, 8))
pollutant_dist = df['pollutant_id'].value_counts()
plt.pie(pollutant_dist, labels=pollutant_dist.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Distribution of Pollutant Types')
plt.show()


# Box plot for pollutant average by state

plt.figure(figsize=(14, 7))
sns.boxplot(x='state', y='pollutant_avg', data=df, palette='coolwarm')
plt.title('Statewise Distribution of Pollutant Averages', fontsize=14)
plt.xlabel('State')
plt.ylabel('Pollutant Avg (μg/m³)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Line plot for pollutant measurements over time
df['year'] = df['last_update'].dt.year
plt.figure(figsize=(12, 6))
pollutant_per_year = df.groupby('year')['pollutant_avg'].mean()
plt.plot(pollutant_per_year.index, pollutant_per_year.values, marker='o')
plt.title('Average Pollutant Levels Over Time')
plt.xlabel('Year')
plt.ylabel('Average Pollutant Level')
plt.grid(True)
plt.show()


# Correlation heatmap for numeric columns
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numeric Columns')
plt.show()

# Step 5: Z-Score Calculation
print("\n--- Z-Score Analysis ---")
# Calculate Z-scores for pollutant_avg to identify outliers
df['z_score_pollutant_avg'] = stats.zscore(df['pollutant_avg'])
print("\nZ-Scores for Pollutant Average:\n", df[['station', 'pollutant_avg', 'z_score_pollutant_avg']].head(10))
# Identify outliers (e.g., Z-score > 3 or < -3)
outliers = df[(df['z_score_pollutant_avg'] > 3) | (df['z_score_pollutant_avg'] < -3)]
print("\nNumber of outliers based on Z-score (>3 or <-3):", len(outliers))
print("\nOutliers:\n", outliers[['station', 'pollutant_avg', 'z_score_pollutant_avg']])

# Step 6: Z-Test
print("\n--- Z-Test ---")
# Example: Z-test to compare pollutant_avg between two states (e.g., Kerala vs Delhi)
from statsmodels.stats.weightstats import ztest
kerala_data = df[df['state'] == 'Kerala']['pollutant_avg'].dropna()
delhi_data = df[df['state'] == 'Delhi']['pollutant_avg'].dropna()

if len(kerala_data) > 0 and len(delhi_data) > 0:
    z_stat, p_val = ztest(kerala_data, delhi_data, value=0)
    print(f"\nZ-test Results (Kerala vs Delhi):")
    print(f"Z-statistic: {z_stat:.3f}")
    print(f"P-value: {p_val:.3f}")
    if p_val < 0.05:
        print("Reject the null hypothesis: Significant difference in pollutant averages between Kerala and Delhi.")
    else:
        print("Fail to reject the null hypothesis: No significant difference in pollutant averages.")
else:
    print("Error: Insufficient data for Z-test between Kerala and Delhi.")

# Step 7: Save the Cleaned Data
df.to_csv('cleaned_air_quality_data.csv', index=False)
print("\nCleaned data saved to 'cleaned_air_quality_data.csv'")

print("\nAnalysis completed!")