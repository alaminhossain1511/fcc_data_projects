import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import data
df =pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = df['weight']/(df['height']/100)**2
for i in range(0,len(df['overweight'])):
  if df['overweight'][i]>25:
    df['overweight'][i]=1
  else:
    df['overweight'][i]=0

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
for i in range(0,len(df['gluc'])):
  if df['gluc'][i]==1:
    df['gluc'][i]=0
  else:
    df['gluc'][i]=1
for i in range(0,len(df['cholesterol'])):
  if df['cholesterol'][i]==1:
    df['cholesterol'][i]=0
  else:
    df['cholesterol'][i]=1

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])['value'].count().to_frame()
    df_cat.rename(columns={'value': 'total'}, inplace=True)
    df_cat.reset_index(inplace=True)

    # Draw the catplot with 'sns.catplot()'
    catplot = sns.catplot(x='variable', y='total', hue='value', col='cardio', kind='bar', data=df_cat)
    fig = catplot.fig


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

# Draw Heat Map
def draw_heat_map():
    # Clean the data
    h25 = df['height'].quantile(0.025)
    h975 = df['height'].quantile(0.975)
    w25 = df['weight'].quantile(0.025)
    w975 = df['weight'].quantile(0.975)
    df_c = df[df['ap_lo'] <= df['ap_hi']]
    df_c = df_c[df_c['height'] >= h25]
    df_c = df_c[df_c['height'] <= h975]
    df_c = df_c[df_c['weight'] >= w25]
    df_c = df_c[df_c['weight'] <= w975]
    df_heat = df_c

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(corr)

    fig, ax = plt.subplots()

    # Draw the heatmap with 'sns.heatmap()'
    heat_map = sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", square=True)
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=90)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig