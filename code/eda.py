import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.model_selection import train_test_split
from itertools import chain
import ast
from itertools import combinations
from collections import Counter, defaultdict


os.chdir("../ds_207_final_project/data/processed")
initial_2024 = pd.read_csv('final_merged_2024.csv')
initial_2025 = pd.read_csv('final_merged_2025.csv')
pd.set_option('display.max_colwidth', None)  # Show full content in a column
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)         # Don't wrap lines early
pd.set_option('display.max_rows', None)      # Optional: show all rows too

# Initialize list to hold extracted information
rows_2024 = []

# Iterate through columns and extract information
for col in initial_2024.columns:
    row = {
        'Column Name': col,                                 # Name of column
        'Data Type': initial_2024[col].dtype,                  # Check data type
        'NaN Count': initial_2024[col].isna().sum(),           # Check NaN counts
        'Unique Values': initial_2024[col].unique().tolist()   # Check unique values
    }

    rows_2024.append(row)

# Convert list to DF and view
nan_summary_2024 = pd.DataFrame(rows_2024)
print("\nTraining and Validation Dataset Summary:") # Print custom summary
nan_summary_2024

# On test data
# Initialize list to hold extracted information
rows_2025 = []

# Iterate through columns and extract information
for col in initial_2025.columns:
    row = {
        'Column Name': col,                                 # Name of column
        'Data Type': initial_2025[col].dtype,                  # Check data type
        'NaN Count': initial_2025[col].isna().sum(),           # Check NaN counts
        'Unique Values': initial_2025[col].unique().tolist()   # Check unique values
    }

    rows_2025.append(row)

# Convert list to DF and view
nan_summary_2025 = pd.DataFrame(rows_2025)
print("\nTest Dataset Summary:") # Print custom summary
nan_summary_2025


# Remove rows with NaN values in feature columns
# Check initial sums
print('BEFORE REMOVING NAN VALUES:')
print('Shape of data:')
print(f'Train/Val: {initial_2024.shape}')
print(f'Test: {initial_2025.shape}')
print('NaN count:')
print(f'{initial_2024.isna().sum()}')
print(f'{initial_2025.isna().sum()}')

df_2024 = initial_2024.copy() # Create copy for processed df
df_2025 = initial_2025.copy() # Create copy for processed df

# Remove rows with NaN values in the listed features
nan_features = ['MovementPrecCollDescription',
                'AirbagDescription',
                'SafetyEquipmentDescription',
                'SobrietyDrugPhysicalDescription1',
                'SpecialInformation',
                'SpeedLimit']
for i in nan_features:
    df_2024 = df_2024.dropna(subset=[i])
    df_2025 = df_2025.dropna(subset=[i])

# Replace remaining NaN values in outcome label to "No Injury"
df_2024['ExtentOfInjuryCode'] = df_2024['ExtentOfInjuryCode'].fillna('No Injury')
df_2025['ExtentOfInjuryCode'] = df_2025['ExtentOfInjuryCode'].fillna('No Injury')


# Check sums after removal
print('\nAFTER REMOVING NAN VALUES:')
print('Shape of data:')
print(f'Train/Val: {df_2024.shape}')
print(f'Test: {df_2025.shape}')
print('NaN count:')
print(f'{df_2024.isna().sum()}')
print(f'{df_2025.isna().sum()}')

# Split into training, validation, and test sets
train_df, val_df = train_test_split(df_2024, test_size=0.2, random_state=42) # 80% train, 20% validation
test_df = df_2025.copy()

# Define numeric and categorical variables
numeric_vars = ['SpeedLimit'] # Notice: Collision ID not normalized
categorical_vars = ['CollisionTypeDescription', 'IsHighwayRelated',
       'Weather1', 'RoadCondition1', 'LightingDescription',
       'ExtentOfInjuryCode', 'MovementPrecCollDescription',
       'AirbagDescription', 'SafetyEquipmentDescription',
       'SobrietyDrugPhysicalDescription1', 'SpecialInformation']

# Drop collision ID
for df in [train_df, val_df, test_df]:
    df.drop(columns=['CollisionId'], inplace=True)

# Normalize numeric variables (speed limit) with training df statistics
speed_mean = train_df['SpeedLimit'].mean()
speed_std = train_df['SpeedLimit'].std()

for df in [train_df, val_df, test_df]:
    df['SpeedLimit'] = (df['SpeedLimit'] - speed_mean) / speed_std

injury_order = ['No Injury', 'Minor', 'Serious', 'Fatal']

for df in [train_df, val_df, test_df]:
  train_df['ExtentOfInjuryCode'] = pd.Categorical(
    train_df['ExtentOfInjuryCode'],
    categories=injury_order,
    ordered=True
    )
  
# Separate categorical features into groups based on whether its a list or not
clear_vars = [
    'CollisionTypeDescription', 'IsHighwayRelated',
    'Weather1', 'RoadCondition1', 'LightingDescription'
]

list_style_vars = [
    'MovementPrecCollDescription', 'AirbagDescription',
    'SafetyEquipmentDescription', 'SobrietyDrugPhysicalDescription1',
    'SpecialInformation'
]

all_vars = clear_vars + list_style_vars # combine into a single variable

train_df['ExtentOfInjuryCode'] = pd.Categorical(
    train_df['ExtentOfInjuryCode'],
    categories=injury_order,
    ordered=True
)

# Iterate through all input variables
for col in all_vars:
    # 1 - Initialize a dictionary to hold (category, injury extent) pairs
    count_dict = defaultdict(Counter)

    # 2 - Iterate through feature values and output values to gather counts
    # Logic for list-style vars
    if col in list_style_vars:
        for _, row in train_df[[col, 'ExtentOfInjuryCode']].iterrows():
            val = row[col]
            injury = row['ExtentOfInjuryCode']

            if pd.isna(val) or pd.isna(injury):
                continue

            try:
                parsed = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                continue

            if isinstance(parsed, list):
                for v in parsed:
                    if isinstance(v, (str, int, float)):
                        count_dict[v][injury] += 1
            elif isinstance(parsed, (str, int, float)):
                count_dict[parsed][injury] += 1

    # Logic for normal (clear) vars
    else:
        for _, row in train_df[[col, 'ExtentOfInjuryCode']].iterrows():
            values = row[col]
            injury = row['ExtentOfInjuryCode']

            if isinstance(values, list):
                for val in values:
                    if isinstance(val, (str, int, float)):
                        count_dict[val][injury] += 1
            elif isinstance(values, (str, int, float)):
                count_dict[values][injury] += 1


    # Get top 10 most common categories
    total_counts = {k: sum(v.values()) for k, v in count_dict.items()}
    top_10 = sorted(total_counts, key=total_counts.get, reverse=True)[:10]

    # Prepare DataFrame
    chart_data = []
    for cat in top_10:
        for injury, count in count_dict[cat].items():
            chart_data.append({
                'Category': cat,
                'ExtentOfInjuryCode': injury,
                'Count': count
            })


    df_chart = pd.DataFrame(chart_data)
    df_chart['ExtentOfInjuryCode'] = pd.Categorical(
        df_chart['ExtentOfInjuryCode'],
        categories=injury_order,
        ordered=True
      )

    # Plot
    base = alt.Chart(df_chart).encode(
        x=alt.X('Category:N', sort='-y'),
        y='Count:Q',
        color=alt.Color('ExtentOfInjuryCode:N', sort=injury_order)
    )

    final_chart = (base.mark_bar() + base.mark_text(
        align='left',
        baseline='bottom',
        dx=10,
        fontSize=10,
        angle=270
    ).encode(
        text=alt.Text('Count:Q')
    )).facet(
        column=alt.Column('ExtentOfInjuryCode:N',
                          sort=injury_order,
                          header=alt.Header(labelAngle=0))
    ).properties(
        title=f"'{col}' by Extent of Injury (Top 10 Categories)"
    )
