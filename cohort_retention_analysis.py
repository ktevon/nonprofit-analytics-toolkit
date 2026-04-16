import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Import dataset
data_types = {
    'contact_id': 'object'
}

df_contacts = pd.read_csv("df_contacts.csv", dtype=data_types, index_col=False)
df_opps = pd.read_csv("df_opps_v5.csv", dtype=data_types, index_col=False)

# index_col=False not working
df_contacts = df_contacts.loc[:, ~df_contacts.columns.str.startswith('Unnamed:')]
df_opps = df_opps.loc[:, ~df_opps.columns.str.startswith('Unnamed:')]

df_opps['close_date'] = pd.to_datetime(df_opps['close_date'], format='ISO8601') # format argument required to silence the error

print(df_contacts.head())
print(df_contacts.info())

print(df_opps.head())
print(df_opps.info())

# Remove regular giving in preparation
df_opps_filtered = df_opps[df_opps['campaign'] != 'Regular Giving'].copy()

print(df_opps_filtered.head())
print(df_opps_filtered.info())

# ---Retention rate by year---

# 1. Get unique year-donor pairs
df_opps_filtered['year'] = df_opps_filtered['close_date'].dt.year
df_years = df_opps_filtered[['contact_id', 'year']].copy()
df_years = df_years.drop_duplicates()

# 2. Identify if the donor gave in the previous year
# Sort by donor and year, then check if the previous row is the same donor 
# AND exactly one year prior.
df_years = df_years.sort_values(['contact_id', 'year'])
df_years['prev_year'] = df_years.groupby('contact_id')['year'].shift(1)
df_years['is_retained'] = (df_years['year'] == df_years['prev_year'] + 1)

# 3. Aggregate results
results = df_years.groupby('year').agg(
    total_donors=('contact_id', 'count'),
    retained_donors=('is_retained', 'sum')
).reset_index()

# 4. Calculate rate - Shift total_donors to get "last year" count for the denominator
results['donors_last_year'] = results['total_donors'].shift(1)
results['retention_rate'] = results['retained_donors'] / results['donors_last_year']

# 5. Clean up - Remove the first year since it has no "previous year" to compare to
df_ret = results.dropna(subset=['donors_last_year']).copy()

print(df_ret)

# ---Second gift rate by year---

# 1. Sort
df_sorted = df_opps_filtered.sort_values(['contact_id', 'close_date'])

# 2. Get the first and second gift dates per contact
first_and_second_gifts = df_sorted.groupby('contact_id')['close_date'].agg([
    ('first_gift_date', 'first'),
    ('second_gift_date', lambda x: x.iloc[1] if len(x) > 1 else pd.NaT)
])

# 3. Identify first gift year
first_and_second_gifts['first_gift_year'] = first_and_second_gifts['first_gift_date'].dt.year

# 4. Calculate months elapsed
delta = first_and_second_gifts['second_gift_date'] - first_and_second_gifts['first_gift_date']
first_and_second_gifts['months_lapsed'] = delta.dt.days / 30.4375 # Divide by 30.4375 to get the number of months

# 5. Flag "Converted" (Second gift within 12 months)
first_and_second_gifts['is_converted'] = first_and_second_gifts['months_lapsed'] <= 12

# 6. Group by year to see the trend
# I exclude 2025 because those donors haven't all had 12 months to recur yet
second_gift_by_year = first_and_second_gifts[first_and_second_gifts['first_gift_year'] < 2025].groupby('first_gift_year').agg(
    total_new_donors=('is_converted', 'count'),
    second_gift_conversions=('is_converted', 'sum')
)

# 7. Calculate the conversion rate
second_gift_by_year['conversion_rate'] = (second_gift_by_year['second_gift_conversions'] / second_gift_by_year['total_new_donors']) * 100

print(second_gift_by_year)

# --- Cohort Analysis---

# 1. Prepare the cohort definitions
cohort_map = first_and_second_gifts['first_gift_year'].to_dict()

# 2. Group by year and contact_id
df_opps_filtered_summary = df_opps_filtered.groupby(['year', 'contact_id']).agg(
    total_amount = ('amount', 'sum')
).reset_index()

# 3. Map the cohort year to the main transactions dataframe
df_opps_filtered_summary['cohort_year'] = df_opps_filtered_summary['contact_id'].map(cohort_map)

# 4. Calculate 'year_number' (Years since first gift)
df_opps_filtered_summary['year_number'] = df_opps_filtered_summary['year'] - df_opps_filtered_summary['cohort_year']

# 5. Group by cohort and year_number to get counts
cohort_counts = df_opps_filtered_summary.groupby(['cohort_year', 'year_number']).agg(
    retained_donors = ('contact_id', 'count'),
    total_amount = ('total_amount', 'sum')
).reset_index()

# 5. Get initial cohort sizes for the denominator
cohort_sizes = cohort_counts[cohort_counts['year_number'] == 0][['cohort_year', 'retained_donors']]
cohort_sizes = cohort_sizes.rename(columns={'retained_donors': 'original_cohort_size'})

# 6. Merge and calculate rate
df_cohorts = cohort_counts.merge(cohort_sizes, on='cohort_year')
df_cohorts['retention_rate'] = df_cohorts['retained_donors'] / df_cohorts['original_cohort_size']

print(df_cohorts)

# "Spaghetti" plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_cohorts, 
    x='year_number', 
    y='retention_rate', 
    hue='cohort_year', 
    palette='viridis', # Colors cohorts from oldest to newest
    marker='o'
)

plt.title('Retention Decay by Cohort Year')
plt.ylabel('Retention Rate (1.0 = 100%)')
plt.xlabel('Years Since First Gift')
plt.grid(True, alpha=0.3)
plt.show()

# Retention heatmap (triangle)
# Pivot the data
cohort_pivot = df_cohorts.pivot(index='cohort_year', columns='year_number', values='retention_rate')

plt.figure(figsize=(12, 8))
sns.heatmap(cohort_pivot, annot=True, fmt=".0%", cmap="YlGnBu", cbar=False)
plt.title('Donor Retention Cohort Analysis', pad=20)
plt.xlabel('Year Number')
plt.ylabel('Cohort Year')
plt.show()

# "Year 1" comparision
# Filter for just the first year of retention
year_1_retention = df_cohorts[df_cohorts['year_number'] == 1]

plt.figure(figsize=(10, 5))
sns.barplot(data=year_1_retention, x='cohort_year', y='retention_rate', palette='Blues_d')
plt.title('Year 1 Retention Rate by Cohort', pad=20)
plt.xlabel('Cohort Year')
plt.ylabel('Retention Rate')
plt.axhline(year_1_retention['retention_rate'].mean(), color='red', linestyle='--', label='Average')
plt.show()

# Visualizing the "revenue mix" for 2025
# Filter for the most recent year
df_2025 = df_cohorts[df_cohorts['cohort_year'] + df_cohorts['year_number'] == 2025].copy()

# Calculate % contribution
total_2025_amt = df_2025['total_amount'].sum()
df_2025['pct_of_total'] = (df_2025['total_amount'] / total_2025_amt) * 100

# Plotting the reliance
plt.figure(figsize=(12, 6))
sns.barplot(data=df_2025, x='cohort_year', y='pct_of_total', palette='mako')
plt.title('Reliance Check: Which Cohorts Funded 2025?', pad=20)
plt.ylabel('% of Total 2025 Income')
plt.xlabel('Donor Recruitment Year')
plt.show()

# Check if 2024/2025 just have 'bigger' donors or just 'more' donors
df_2025['avg_gift'] = df_2025['total_amount'] / df_2025['retained_donors']
print(df_2025[['cohort_year', 'retained_donors', 'avg_gift']])

# Check median to see if a handful of donors are skewing the average
df_2025_medians = df_opps_filtered_summary[df_opps_filtered_summary['year'] == 2025].groupby('cohort_year')['total_amount'].median()
print(df_2025_medians)

# Box plot for 2024 cohort who donated in 2025
cohort_2024_in_2025 = df_opps_filtered_summary[(df_opps_filtered_summary['cohort_year'] == 2024) & (df_opps_filtered_summary['year'] == 2025)].copy()

print(cohort_2024_in_2025)

sns.boxplot(data=cohort_2024_in_2025, y='total_amount')
plt.show()

# Do donors give more as they age?
df_cohorts['avg_gift_per_donor'] = df_cohorts['total_amount'] / df_cohorts['retained_donors']

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_cohorts, x='year_number', y='avg_gift_per_donor', hue='cohort_year', legend=None)
plt.title('Do Donors Give More as They Age?')
plt.ylabel('Average Annual Amount per Donor')
plt.xlabel('Years Since First Gift')
plt.show()