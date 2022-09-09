import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)
df.drop(columns='BMI', inplace=True)


### Normalize data by making 0 always good and 1 always bad. 

# If the value of 'cholesterol' or 'gluc' is 1, make the value 0. 
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['gluc'] == 1, 'gluc'] = 0

# If the value is more than 1, make the value 1.
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df.loc[df['gluc'] > 1, 'gluc'] = 1


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature
    # You will have to rename one of the columns for the catplot to work correctly
    to_plot = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).value_counts()
    to_plot = to_plot.rename(columns={'count': 'total'})

    # Draw the catplot with 'sns.catplot()' (returns a FacetGrid)
    g = sns.catplot(data=to_plot, x='variable', y='total', kind='bar', col='cardio', hue='value')
    fig = g.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data

    # diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
    pressure_mask = df['ap_lo'] <= df['ap_hi']
    pressure_mask.value_counts()

    # height is less than the 2.5th percentile (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
    # height is more than the 97.5th percentile
    height_mask = (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975))
    height_mask.value_counts()

    # weight is less than the 2.5th percentile
    # weight is more than the 97.5th percentile
    weight_mask = (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))
    weight_mask.value_counts()

    df_heat = df[pressure_mask & height_mask & weight_mask]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(9,6))

    # Draw the heatmap with 'sns.heatmap()'
    ax = sns.heatmap(corr, mask=mask, linewidths=0.5, annot=True, fmt='.1f', cmap='mako')

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
