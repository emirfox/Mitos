import pandas as pd
import numpy as np
from sklearn.utils import resample

## equal number of division, new csv


df=pd.read_csv(r'C:\Users\HP\Desktop\staj\sentiment_data\EcoPreprocessed.csv')

# Find the minimum number of divison
min_num = df['division'].value_counts().min()

df_positive = df[df['division']=='positive']
df_negative = df[df['division']=='negative']
df_neutral = df[df['division']=='neutral']


df_positive_downsampled = resample(df_positive,
                                 replace=False,    # sample without replacement
                                 n_samples=min_num, # to match minority class
                                 random_state=123) # reproducible results

df_negative_downsampled = resample(df_negative,
                                 replace=False,
                                 n_samples=min_num,
                                 random_state=123)

df_neutral_downsampled = resample(df_neutral,
                                 replace=False,
                                 n_samples=min_num,
                                 random_state=123)

# Combine minority and downsampled majority
df_downsampled = pd.concat([df_positive_downsampled, df_negative_downsampled, df_neutral_downsampled])

# Check counts
print(df_downsampled['division'].value_counts())
df_downsampled.to_csv('balanced_division.csv', index=False)