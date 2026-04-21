import pandas as pd
from matplotlib import pyplot as plt

# Import dataset
data_types = {
    'contact_id': 'object'
}

df_giw = pd.read_csv("C:/Users/ktevo/OneDrive/Medium/3&4 Will My Synthetic Donors Leave a Gift/df_giw_2026-04-02.csv", dtype=data_types, index_col=False)
df_opps = pd.read_csv("C:/Users/ktevo/OneDrive/Medium/2 The Day My Synthetic Donors Didn't Pass for Human/df_opps_v5.csv", dtype=data_types, index_col=False)

# index_col=False not working for df_opps
# df_giw = df_giw.loc[:, ~df_giw.columns.str.startswith('Unnamed:')]
df_opps = df_opps.loc[:, ~df_opps.columns.str.startswith('Unnamed:')]

df_opps['close_date'] = pd.to_datetime(df_opps['close_date'], format='ISO8601') # format argument required to silence the error

print(df_giw.head())
print(df_opps.head())

print(df_giw.info())
print(df_opps.info())

# Remove regular giving in preparation
df_opps_filtered = df_opps[df_opps['campaign'] != 'Regular Giving'].copy()

print(df_opps_filtered.head())
print(df_opps_filtered.info())

# ---Historical Amount and Original Cohort Size---

# 1. Add a 'year' column
df_opps_filtered['year'] = df_opps_filtered['close_date'].dt.year

# 2. Sort
df_sorted = df_opps_filtered.sort_values(['contact_id', 'close_date'])

# 3. Get the first date per contact
first_gifts = df_sorted.groupby('contact_id')['close_date'].agg([
    ('first_gift_date', 'first')
])

# 4. Identify first gift year
first_gifts['first_gift_year'] = first_gifts['first_gift_date'].dt.year

# 5. Prepare the cohort definitions
cohort_map = first_gifts['first_gift_year'].to_dict()

# 6. Group by year and contact_id
df_opps_filtered_summary = df_opps_filtered.groupby(['year', 'contact_id']).agg(
    total_amount = ('amount', 'sum')
).reset_index()

# 7. Map the cohort year to the main transactions dataframe
df_opps_filtered_summary['cohort_year'] = df_opps_filtered_summary['contact_id'].map(cohort_map)

# 8. Calculate 'year_number' (Years since first gift)
df_opps_filtered_summary['year_number'] = df_opps_filtered_summary['year'] - df_opps_filtered_summary['cohort_year']

# 9. Group by cohort and year_number to get counts
cohort_counts = df_opps_filtered_summary.groupby(['cohort_year', 'year_number']).agg(
    retained_donors = ('contact_id', 'count'),
    total_amount = ('total_amount', 'sum')
).reset_index()

# 10. Get initial cohort sizes for the denominator
cohort_sizes = cohort_counts[cohort_counts['year_number'] == 0][['cohort_year', 'retained_donors']]
cohort_sizes = cohort_sizes.rename(columns={'retained_donors': 'original_cohort_size'})

# 11. Merge and calculate rate
df_cohorts = cohort_counts.merge(cohort_sizes, on='cohort_year')
df_cohorts['retention_rate'] = df_cohorts['retained_donors'] / df_cohorts['original_cohort_size']

print(df_cohorts.head())

# 1. Historical cash (Already in the bank)
df_ltv = df_cohorts.groupby('cohort_year').agg(
    historical_amt=('total_amount', 'sum'),
    original_size=('original_cohort_size', 'first')
).reset_index()

print(df_ltv)

# ---Future Amount---

# Get the count of donors currently active in 2025 (for future cash)
# Filter for the most recent year
df_2025 = df_cohorts[df_cohorts['cohort_year'] + df_cohorts['year_number'] == 2025].copy()

# Add active_average_gifts
df_2025['avg_gift_2025'] = df_2025['total_amount'] / df_2025['retained_donors']

# Merge into df_ltv
df_ltv = df_ltv.merge(df_2025, how='left', on='cohort_year')
print(df_ltv)

# Retention rate
# Sort to ensure sequential years
df_cohorts = df_cohorts.sort_values(['cohort_year', 'year_number'])

# Calculate the year-over-year retention
df_cohorts['prev_year_donors'] = df_cohorts.groupby('cohort_year')['retained_donors'].shift(1)
df_cohorts['annual_retention'] = df_cohorts['retained_donors'] / df_cohorts['prev_year_donors']

# Calculate a 3-year rolling average of retention per cohort
df_cohorts['smoothed_r'] = df_cohorts.groupby('cohort_year')['annual_retention'].transform(lambda x: x.rolling(3, min_periods=1).mean())

# For the LTV calculation, take the 'annual retention' of 2025, or the maximum 'year_number'
current_annual_r = df_cohorts[df_cohorts['year_number'] == df_cohorts.groupby('cohort_year')['year_number'].transform('max')][['cohort_year', 'smoothed_r']]

# Cap the smoothed value
current_annual_r['r_for_ltv'] = current_annual_r['smoothed_r'].clip(upper=0.95)
print(current_annual_r)

# Append 'current_annual_r'
df_ltv = df_ltv.merge(current_annual_r[['cohort_year', 'r_for_ltv']], how='left', on='cohort_year')

# Estimate future cash
R = df_ltv['r_for_ltv']
df_ltv['expected_remaining_years'] = R / (1 - R)

# Calculate Future Amount
df_ltv['future_amount'] = df_ltv['retained_donors'] * df_ltv['avg_gift_2025'] * df_ltv['expected_remaining_years']

print(df_ltv)

# ---Bequest Estimated Value---

# Set the average bequest value
avg_bequest_value = 50000

def calculate_bequest_probability(age, tenure):
    # Base probability
    base_p = 0.4 
    
    # Age factor: older donors are more 'certain' realizations
    age_factor = 1.2 if age > 70 else 0.8
    
    # Tenure factor: long-term loyalty increases 'will-stickiness'
    tenure_factor = 1.3 if tenure > 60 else 0.9 # Tenure greater than 60 months or 5 years
    
    return base_p * age_factor * tenure_factor

# Apply to confirmed donors
df_giw['bequest_p'] = df_giw.apply(
    lambda row: calculate_bequest_probability(row['age'], row['tenure']) 
    if row['bequest_status'] == 'Confirmed' else 0, 
    axis=1
)

df_giw['bequest_ev'] = df_giw['bequest_p'] * avg_bequest_value

# Append 'cohort_year'
df_giw['cohort_year'] = df_giw['contact_id'].map(cohort_map)

# Calculate LTV by cohort
bequest_ltv = df_giw.groupby('cohort_year')['bequest_ev'].sum()

# Merge into df_ltv
df_ltv = df_ltv.merge(bequest_ltv, how='left', on='cohort_year')

print(df_ltv)

# Final Comprehensive LTV calculation
df_ltv['comp_ltv'] = (
    df_ltv['historical_amt'] + 
    df_ltv['future_amount']+ 
    df_ltv['bequest_ev']
) / df_ltv['original_size']

# Remove 2025
df_ltv = df_ltv[df_ltv['cohort_year'] != 2025]

print(df_ltv)

# ---Comprehensive LTV stacked bar chart ---

# Select necessary columns only
df_ltv_pct = df_ltv[['cohort_year', 'historical_amt', 'future_amount', 'bequest_ev']].copy()

# Define the columns for the stack
cols = ['historical_amt', 'future_amount', 'bequest_ev']

# Calculate the row-wise sum
total = df_ltv_pct[cols].sum(axis=1)

# 3. Divide the columns by the total and assign to new names
# This creates 'historical_amt_pct', 'future_amount_pct', etc.
df_ltv_pct[[f"{c}_pct" for c in cols]] = df_ltv_pct[cols].div(total, axis=0)

print(df_ltv_pct)

# ---Comprehensive LTV stacked bar chart---

# Select necessary columns only
df_ltv_pct = df_ltv[['cohort_year', 'historical_amt', 'future_amount', 'bequest_ev']].copy()

# Define the columns to transform
cols_to_div = ['historical_amt', 'future_amount', 'bequest_ev']

# Calculate the row-wise sum
total = df_ltv_pct[cols_to_div].sum(axis=1)

# Divide the columns by the total and assign to new names
# This creates 'historical_amt_pct', 'future_amount_pct', etc.
df_ltv_pct[[f"{c}_pct" for c in cols_to_div]] = df_ltv_pct[cols_to_div].div(total, axis=0)

# Select necessary columns only and set index
df_ltv_pct = df_ltv_pct[['cohort_year', 'historical_amt_pct', 'future_amount_pct', 'bequest_ev_pct']].copy().set_index('cohort_year')

# Plot
df_ltv_pct.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#264653', '#457B9D', '#2A9D8F'])
plt.title('Composition of LTV by Cohort Year', fontsize=14, pad=20)
plt.ylabel('Percentage (%)')
plt.xlabel('Cohort Year')
plt.legend(title="Components", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()