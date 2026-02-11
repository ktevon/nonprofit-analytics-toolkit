# Import required modules
import pandas as pd
import numpy as np
from faker import Faker
from datetime import date
import random
import matplotlib.pyplot as plt

# Instantiate Faker
fake = Faker()

# Set seed for reproducibility
Faker.seed(42) 
random.seed(42)
np.random.seed(42)

# Configuration
num_contacts = 5000
num_adhoc_opps = 18000 # 50000 in the original dataset
start_date = date(2016, 1, 1) # January 1, 2021 in the original dataset
end_date = date(2025, 12, 31)
years = list(range(start_date.year, end_date.year + 1, 1))

# --- Create Contacts ---
def generate_contacts_with_signals(n):
    """Generate a contact dataframe with age signals"""

    contacts = [] # Initialize a contacts list
    
    for i in range(n):

        # Generate basic demographics
        age = random.randint(18, 90)
        gender = np.random.choice(['F', 'M', 'Non-binary'], p=[0.52, 0.45, 0.03])
        name = fake.name_male() if gender == 'M' else fake.name_female() if gender == 'F' else fake.name_nonbinary()
        
        # Bake in the signal:
        # Make "Major Donor" status dependent on age
        major_prob = 0.001 + (age / 100) * 0.05
        is_major = random.random() < major_prob
        
        # Make "Regular Donor" status inversely dependent on age
        reg_prob = 0.40 - (age / 100) * 0.30
        is_regular = random.random() < reg_prob

        contacts.append({
            'contact_id': f"003{str(i+1).zfill(12)}",
            'age': age,
            'gender': gender,
            'name': name,
            'is_major': is_major,
            'is_regular': is_regular
        })

    return pd.DataFrame(contacts)

df_contacts = generate_contacts_with_signals(num_contacts)

print(df_contacts.head())

# --- Generate Regular Donor Contact Records ---

# Define monthly weights
monthly_weights = [0.6, 0.9, 1.0, 1.1, 1.1, 1.2, 1.2, 1.1, 1.0, 1.0, 0.8, 0.4]

# Repeat the monthly_weights for each year to prepare the seasonal probabilities for all 60 months (5 years * 12 months)
all_months_weights = monthly_weights * len(years)

# Convert weights to probabilities
probabilities = np.array(all_months_weights) / sum(all_months_weights)

# Create a list of all possible (Year, Month) starting points

month_options = [] # Initialize a month_options list

for year in years:
    for month in range(1, 13):
        month_options.append((year, month))

# Get a list and number of regular donor IDs
regular_contact_ids = df_contacts[df_contacts['is_regular'] == True]['contact_id'].tolist()
num_regular = len(regular_contact_ids)

# Assign a start month to every single regular donor at once
chosen_start_points = np.random.choice(len(month_options), size=num_regular, p=probabilities)

# Create regular donor records
regular_donors = [] # Initialize a regular_donors list

for i, start_idx in enumerate(chosen_start_points):

    cid = regular_contact_ids[i] # Pick a donor
    year, month = month_options[start_idx] # Pick the first-gift month
    start_dt = date(year, month, 1) # Pick the first-gift date
    
    months_stayed = 1 # Initialize to 1 (the first month donors give)

    # Calculate the number of months left between start and the very end (31 Dec 2025)
    max_possible_months = (end_date.year - start_dt.year) * 12 + (end_date.month - start_dt.month) + 1 # Add 1 at the end
    
    for m in range(2, max_possible_months + 1):
        # The "logarithmic decay": fast at first, slow later
        drop_prob = 0.15 / (1 + np.log(m-1)) 
        if random.random() < drop_prob:
            break # They cancelled!
        months_stayed = m
    
    regular_donors.append({
        'contact_id': cid,
        'start_date': start_dt,
        'months': months_stayed,
        'amount': random.choice(list(range(10, 105, 5)))
    })

# Convert to a data frame to verify
df_regular_donors = pd.DataFrame(regular_donors)

print(f"Total Regular Donors processed: {len(df_regular_donors)}")

# Visualize monthly regular donor acquisitions
df_regular_donors['start_date'].apply(lambda x: x.month).value_counts().sort_index().plot(kind='bar', color="#264653")
plt.title("Monthly Regular Donor Acquisitions", pad=20)
plt.xlabel("Month")
plt.ylabel("Number of Donors")

plt.show()
plt.close()

# --- Generate Regular Donation Transaction Records ---

rg_transactions = [] # Initialize an rg_transactions list

for donor in regular_donors:
    
    for m in range(donor['months']):
        # Increment month
        tx_date = donor['start_date'] + pd.DateOffset(months=m)

        if tx_date.date() <= end_date:
            # Generate the ID based on how many items are already in the list
            unique_id = f"006_REG_{str(len(rg_transactions)).zfill(8)}"

            rg_transactions.append({
                'opportunity_id': unique_id,
                'contact_id': donor['contact_id'],
                'close_date': tx_date.date(),
                'amount': donor['amount'],
                'stage': 'Closed Won',
                'type': 'Regular'
            })

df_rg = pd.DataFrame(rg_transactions)

print(df_rg.head())

# --- Generate One-off Donations ---

# Configure seasonality
au_seasonal_weights = [1.0, 0.8, 1.0, 1.2, 3.5, 5.0, 1.2, 1.0, 1.1, 1.5, 4.0, 6.0]

def get_weighted_random_date(year):
    """Returns a date within the year based on AU seasonal peaks."""
    month = np.random.choice(range(1, 13), p=np.array(au_seasonal_weights)/sum(au_seasonal_weights))
    
    # Handle month lengths. Ignore leap years
    if month == 2:
        day = random.randint(1, 28)
    elif month in [4, 6, 9, 11]:
        day = random.randint(1, 30)
    else:
        day = random.randint(1, 31)
        
    return date(year, month, day)

# Define rounding logic
def round_amount(amount, base):
    """Rounds an amount and returns an integer that is a multiple of the base number"""
    return int(base * round(amount/base))

# Assign acquisition years to spread the 5,000 contacts across all years
contact_ids = df_contacts["contact_id"].tolist()
acq_date = {cid: get_weighted_random_date(np.random.choice(years)) for cid in contact_ids}

# Expand the pool of people capable of giving more than once
repeat_pool_size = int(num_contacts * 0.45)
repeat_donor_ids = np.random.choice(contact_ids, size=repeat_pool_size, replace=False)

repeat_loyalty_weights = np.random.pareto(a=12.0, size=repeat_pool_size)
repeat_loyalty_weights /= repeat_loyalty_weights.sum()

def generate_segmented_donations_fix3(target_n, df_contacts):
    
    """Generates transaction records for one-off gifts, segmented by donor type"""
    
    donations = [] # Initialize a donations list

    # Pre-map the major donor status for speed
    major_map = df_contacts.set_index('contact_id')['is_major'].to_dict()

    for i in range(target_n):

        # Determiine donor and date
        if i < num_contacts: # Acquisition logic
            donor_id = contact_ids[i]
            close_date = acq_date[donor_id]
        else: # Loyalty/Repeat logic
            # For the remaining gifts, use the loyalty weights - Only pick the 30% who are capable of repeating
            donor_id = np.random.choice(repeat_donor_ids, p=repeat_loyalty_weights)

            # Calculate how many years are left in the charity's life
            first_year = acq_date[donor_id].year
            years_remaining = [y for y in years if y >= first_year]

            # The fix for the 2024 Spike: Use "Relative Age" weights
            # Probability depends on how many years it has been since their first gift
            weights = []

            for y in years_remaining:
                years_since_acq = y - first_year
                if years_since_acq == 0:
                    weights.append(10) # High chance to give again in the same year
                elif years_since_acq == 1:
                    weights.append(5)  # Moderate chance in Year 1
                elif years_since_acq == 2:
                    weights.append(2)  # Lower chance in Year 2
                else:
                    weights.append(0.5) # The "Long Tail" for Legacy propensity

            weights = np.array(weights) / sum(weights) # Normalize

            # Pick the year using these weights
            close_year = np.random.choice(years_remaining, p=weights)

            # Proceed with your seasonal logic
            close_date = get_weighted_random_date(close_year)

            # Calculate the number of days between acquisition and the end of 2025 - Determine the window of opportunity
            contact_start_date = pd.to_datetime(acq_date[donor_id])
            contact_end_date = pd.to_datetime(end_date)
            days_diff = (contact_end_date - contact_start_date).days

            # # Date Safety Catch
            # If the seasonal date falls before acquisition, pick a random day in their remaining "tenure"
            if pd.to_datetime(close_date) < contact_start_date:
                if days_diff > 0:
                    # Pick any day between day 1 and end_date
                    random_days = random.randint(1, days_diff)
                    close_date = contact_start_date + pd.Timedelta(days=random_days)
                else:
                    # For donors joined on the very last day
                    close_date = contact_start_date

        # Check if this donor is 'Major'
        if major_map[donor_id]:
            # --- Major Donor Logic ---
            # Lower alpha (1.2) = More extreme variance
            # Offset (+1000) = Minimum major gift is $1k
            amount = np.random.pareto(a=1.2) * 500 + 1000
            # Cap at $50k so one person doesn't ruin the charity's budget!
            if amount > 50000: amount = np.random.uniform(20000, 50000)
        else:
            # --- General Donor Logic ---
            # Higher alpha (3.0) = Tighter, more predictable gifts
            amount = np.random.pareto(a=3.0) * 75 + 25
            if amount > 1000: amount = np.random.uniform(500, 1000)

        # Rounding logic
        if amount >= 1000:
            amount = round_amount(amount, 500)
        elif amount >= 100:
            amount = round_amount(amount, 50)
        else:
            amount = round_amount(amount, 5)

        donations.append({
            'opportunity_id': f"006_GEN_{str(i).zfill(8)}",
            'contact_id': donor_id,
            'close_date': close_date,
            'amount': amount,
            'is_major_gift': major_map[donor_id],
            'stage': 'Closed Won'
            })
    
    return pd.DataFrame(donations)

df_adhoc = generate_segmented_donations_fix3(num_adhoc_opps, df_contacts)
print(df_adhoc.info())
print(df_adhoc.head())

# Combine df_rg and df_adhoc
df_opps = pd.concat([df_rg, df_adhoc]).reset_index(drop=True)
print(df_opps.head())


# --- Adding campaigns ---
def assign_campaign(row, unsolicited_random_indices):
    # 1. Regular Giving (Highest Priority)
    if row['type'] == 'Regular':
        return 'Regular Giving'
    
    # 2. Random 5% Unsolicited
    if row.name in unsolicited_random_indices:
        return 'Unsolicited'
    
    # 3. Major Giving
    if row['is_major_gift'] == True:
        return 'Major Giving'
    
    # Date-based logic for the rest
    d = row['close_date']
    m, day = d.month, d.day
    
    # 4. Tax Appeal: 1 May to 15 July
    if (m == 5) or (m == 6) or (m == 7 and day <= 15):
        return 'Tax Appeal'
    
    # 5. Christmas Appeal: 1 Nov to 15 Jan
    if (m == 11) or (m == 12) or (m == 1 and day <= 15):
        return 'Christmas Appeal'
    
    # 6. Spring Newsletter: 16 Sep to 15 Oct
    if (m == 9 and day >= 16) or (m == 10 and day <= 15):
        return 'Spring Newsletter'
    
    # 7. Autumn Newsletter: 16 Mar to 15 Apr
    if (m == 3 and day >= 16) or (m == 4 and day <= 15):
        return 'Autumn Newsletter'
    
    # 8. Remaining Unsolicited
    return 'Unsolicited'

# Pre-select the 5% random indices for Step 2 
non_regular_indices = df_opps[df_opps['type'] != 'Regular'].index # Excluding regular donors so I don't overwrite them
unsolicited_sample_size = int(len(df_opps) * 0.05)
unsolicited_indices = np.random.choice(non_regular_indices, unsolicited_sample_size, replace=False)

# Apply the function
df_opps['campaign'] = df_opps.apply(
    lambda x: assign_campaign(x, unsolicited_indices), 
    axis=1
)

# Replace NaN in the is_major_gift column with False
df_opps["is_major_gift"] = df_opps["is_major_gift"].astype("boolean").fillna(False)

print(df_opps.head())