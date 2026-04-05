# Import required modules
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

# Set seed for reproducibility
np.random.seed(42)

# Import dataset
data_types = {
    'contact_id': 'object'
}

df_contacts = pd.read_csv("df_contacts.csv", dtype=data_types, index_col=False)
df_opps = pd.read_csv("df_opps.csv", dtype=data_types, index_col=False)

# Remove the index column if index_col=False doesn't work
df_contacts = df_contacts.loc[:, ~df_contacts.columns.str.startswith('Unnamed:')]
df_opps = df_opps.loc[:, ~df_opps.columns.str.startswith('Unnamed:')]

df_opps['close_date'] = pd.to_datetime(df_opps['close_date'], format='ISO8601').dt.date # format argument required to silence the error

print(df_contacts.info())
print(df_contacts.head())

print(df_opps.info())
print(df_opps.head())

# Identify the maximum date in the dataset
end_date = max(df_opps['close_date'])

print(end_date)

# Create a function to generate RFMT features
def generate_rfmt(data):

    """Generate features for the recency, frequency, monetary value, and tenure (RFMT) of donor activities"""

    # Identeify the last gift date, first gift date, frequency, and monetary value for each donor
    df = data.groupby(['contact_id']).agg({
        'close_date': ["max", "min"],
        'amount': ["count", "sum"]
        })
    
    df.columns = ['last_gift_date', 'first_gift_date', 'frequency', 'monetary_value'] # This works if you know the column order and want to set new names directly

    df['last_gift_date'] = pd.to_datetime(df['last_gift_date']).dt.date
    df['first_gift_date'] = pd.to_datetime(df['first_gift_date']).dt.date

    # Calculate the time elapsed since the last gift in months
    df['recency'] = df['last_gift_date'].apply(lambda d: (relativedelta(end_date, d).years * 12) + relativedelta(end_date, d).months) 
    # Calculate the time between last and first gifts in months
    df['tenure'] = df.apply(lambda row: (relativedelta(row['last_gift_date'], row['first_gift_date']).years * 12) + relativedelta(row['last_gift_date'], row['first_gift_date']).months, axis = 1)

    df = df.drop(['last_gift_date', 'first_gift_date'], axis = 1) # 'last gift date' and 'first gift date' are no longer required

    df = df.reset_index()

    return df

df_rfmt = generate_rfmt(df_opps)

print(df_rfmt.head())

# Create age groups
bins = [0, 39, 49, 59, 69, 90]
labels = ['under_40', '40-49', '50-59', '60-69', '70_or_over']

df_giw = df_contacts.copy()
df_giw['age_group'] = pd.cut(df_giw['age'], bins = bins, labels = labels)

print(df_giw.head())

# Identify regular gift status
df_regular_donors = df_opps[df_opps['type'] == 'Regular'].copy()

# Idenity first and last regular giving date
df_regular_donors = df_regular_donors.groupby('contact_id')['close_date'].agg([
    ('first_rg_date', 'first'),
    ('last_rg_date', lambda x: x.iloc[-1])
]).reset_index()

# Convert to datetime
df_regular_donors['first_rg_date'] = pd.to_datetime(df_regular_donors['first_rg_date'])
df_regular_donors['last_rg_date'] = pd.to_datetime(df_regular_donors['last_rg_date'])

# Add a regular giving status column based on the condition
# Check if the period is exactly December 2025
is_dec_2025 = df_regular_donors['last_rg_date'].dt.to_period('M') == '2025-12'
df_regular_donors['rg_status'] = np.where(is_dec_2025, 'Active', 'Cancelled')

print(df_regular_donors.head())

# Add df_rfmt and df_regular_donors to df_giw
df_giw = pd.merge(df_giw, df_rfmt, how='right', on = 'contact_id') # how = 'right' to exclude contacts with no transaction history
df_giw = pd.merge(df_giw, df_regular_donors[['contact_id', 'rg_status']], how = 'left', on = 'contact_id')

# Replace NaN with 'No RG'
df_giw['rg_status'] = df_giw['rg_status'].fillna('No RG')

# Drop unnecessary columns
df_giw = df_giw.drop(columns = ['gender', 'name', 'is_major', 'is_regular'])

print(df_giw.head())

# Define the "stages of engagement"
# 0-18m (Recent), 18-42m (Sweet Spot), 42-84m (Dormant), 84+ (Anchor)
r_bins = [-1, 18, 42, 84, 1000] 
r_labels = [4, 5, 2, 1] 

df_giw['r_score'] = pd.cut(
    df_giw['recency'], 
    bins=r_bins, 
    labels=r_labels
).astype(int)

# Define the donor archetypes
# 0-2 (One-offs), 3-9 (Occasional), 10-49 (Annual), 50-99 (Chronic), 100+ (Revolutionary)
f_bins = [-1, 2, 9, 49, 99, 10000]
f_labels = [0, 1, 4, 7, 10]

df_giw['f_score'] = pd.cut(
    df_giw['frequency'], 
    bins=f_bins, 
    labels=f_labels
).astype(int)

# Define custom scoring mapping for each quintile (0-20%, 20-40%, etc.)
# Quintile 1 (Bottom 20%) -> Score 0
# Quintile 2 (20-40%)     -> Score 2
# Quintile 3 (40-60%)     -> Score 3 (The Sweet Spot)
# Quintile 4 (60-80%)     -> Score 3 (The Sweet Spot)
# Quintile 5 (Top 20%)    -> Score 1 (The Upper Crust)

# 1. Assign standard quintile ranks (1 through 5)
df_giw['m_quintile'] = pd.qcut(
    df_giw['monetary_value'], 
    q=5, 
    labels=[1, 2, 3, 4, 5],
    duplicates='drop'
)

# 2. Map those ranks to custom scores
score_map = {1: 0, 2: 2, 3: 3, 4: 3, 5: 1}
df_giw['m_score'] = df_giw['m_quintile'].map(score_map).astype(int)

# Calculate 't_score' using weights that reward long-term loyalty more heavily
df_giw['t_score'] = pd.cut(df_giw['tenure'], bins=5, labels=[0, 1, 3, 6, 10]).astype('Int64')

# Map age
age_map = {
    '70_or_over': 10,
    '60-69': 7,
    '50-59': 3,
    '40-49': 1,
    'under_40': 0
}

df_giw['age_score'] = df_giw['age_group'].map(age_map).astype('Int64')

# Map the RG weights
rg_weights = {'Cancelled': 1.2, 'Active': 1.0, 'No RG': 0.5}

df_giw['rg_weight'] = df_giw['rg_status'].map(rg_weights)

print(df_giw.head())

# Calculate the total score
# Give tenure a higher 'weight' here to ensure it survives the model's noise
df_giw['raw_propensity'] = (
    (df_giw['r_score'] * 1.0) + 
    (df_giw['f_score'] * 1.0) +
    (df_giw['m_score'] * 0.5) + 
    (df_giw['t_score'] * 2.5) + # Boost the 'signal' of tenure
    (df_giw['age_score'] * 2.0)   # Age is still important, but not a 'gate'
 ) * df_giw['rg_weight']

# Use raw_propensity to create a probability
# 1. Scale the probability so the 'best' prospects have a 20% chance
max_p = df_giw['raw_propensity'].max()
df_giw['assignment_prob'] = (df_giw['raw_propensity'] / max_p) * 0.20

# 2. Initial Assignment
df_giw['bequest_status'] = np.random.rand(len(df_giw)) < df_giw['assignment_prob']

# 3. The 'surprise' bequest donors (the wildcards)
# Pick 5 random people who are 'NA' and make them 'Confirmed'
wildcards = df_giw[df_giw['bequest_status'] == False].sample(5).index
df_giw.loc[wildcards, 'bequest_status'] = True

# 4. Final Formatting
df_giw['bequest_status'] = df_giw['bequest_status'].map({True: 'Confirmed', False: 'NA'})
print(df_giw.head())
print(df_giw.info())

# See the average profile of bequest donors
print(df_giw[df_giw['bequest_status'] == 'Confirmed'].describe())

# Check the RG status of bequest donors
print(df_giw[df_giw['bequest_status'] == 'Confirmed']['rg_status'].value_counts(normalize=True))