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

# Normalize data by making 0 always good and 1 always bad.

# If the value of 'cholesterol' or 'gluc' is 1, make the value 0.
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['gluc'] == 1, 'gluc'] = 0

# If the value is more than 1, make the value 1.
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df.loc[df['gluc'] > 1, 'gluc'] = 1

# Draw Categorical Plot


def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt`
    # moves the `value_var` columns to rows with their name under `variable` and their value under `value`
    cat_df = pd.melt(df, id_vars=['cardio'], value_vars=[
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group the data to show count totals by each group of cardio, variable, and value
    plot_df = cat_df.groupby(
        ['cardio', 'variable', 'value'], as_index=False).size()

    # rename the resulting `size` column to `total``
    plot_df = plot_df.rename(columns={'size': 'total'})

    # Draw the catplot. 'sns.catplot()' returns a FacetGrid
    g = sns.catplot(data=plot_df, x='variable', y='total',
                    kind='bar', col='cardio', hue='value')
    fig = g.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Filter out the invalid data

    # where diastolic pressure is higher than systolic
    invalid_pressure = df['ap_lo'] > df['ap_hi']

    # height is less than the 2.5th or more than the 97.5th percentile
    invalid_height = (df['height'] < df['height'].quantile(0.025)) | (
        df['height'] > df['height'].quantile(0.975))

    # weight is less than the 2.5th or more than the 97.5th percentile
    invalid_weight = (df['weight'] < df['weight'].quantile(0.025)) | (
        df['weight'] > df['weight'].quantile(0.975))

    # keep only data that is not any of the invalid conditions
    df_heat = df[~(invalid_pressure | invalid_height | invalid_weight)]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr))

    # Draw the heatmap
    fig, ax = plt.subplots(figsize=(9, 6))
    ax = sns.heatmap(corr, mask=mask, linewidths=0.5,
                     annot=True, fmt='.1f', cmap='mako')

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
