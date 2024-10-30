import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset

df = pd.read_csv('medical_examination.csv')

# 1. Add 'BMI' column

df['BMI'] = df['weight'] / (df['height']/100) **2
df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
# 2. Add 'overweight' column

df['overweight'] = df['BMI'].apply(lambda x: 1  if pd.notna(x) and  x > 25 else 0)

# # Normalize data for 'cholesterol' and 'glucose
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if pd.notna(x) and  x == 'normal' else '1')
df['gluc'] = df['gluc'].apply(lambda x: 0 if pd.notna(x) and x == 'normal' else '1')



# 3. Create a categorical plot for cholesterol, glucose, smoking, alcohol, active, and overweight based on cardio (heart disease)

def draw_cat_plot():
    # Melt the dataframe for easier plotting
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alcohol', 'active', 'overweight'])

     # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename
    # one of the collumns for the catplot to work correctly.

    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index()
    
    df_cat = df_cat.rename(columns={0: 'total'})
    df_cat['variable'] = df_cat['variable'].astype(str)
    df_cat['value'] = df_cat['value'].astype(str)
    df_cat['total'] = pd.to_numeric( df_cat['total'], errors='coerce')
    df_cat['cardio'] = pd.to_numeric( df_cat['cardio'], errors='coerce')

    # Draw the catplot using Seaborn
    fig = sns.catplot(x="variable",y='total',  hue="value", col="cardio", data=df_cat, kind="bar", height=5, aspect=1.0)
    
    fig.set_axis_labels('variable', 'total')
    #fig.set_titles('cardiovascular Disease: {}')
    plt.tight_layout()
    plt.show()

draw_cat_plot()

# 4. Draw a heatmap of correlations between features

def draw_heat_map():
    # Clean  the Data
    # Systolic pressure should be higher than diastolic
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

    corr = df_heat.corr()
     # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure

    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw the heatmap with Seaborn

    sns.heatmap(corr, annot=True, mask=mask, fmt=' .1f', center=0, linewidths=0.5, cbar_kws={'shrink': 0.5}, ax=ax)

    plt.tight_layout()
    plt.show()

draw_heat_map()
