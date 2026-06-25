"""
features.py
============
BEGINNER EXPLANATION:
This file takes your cleaned dataset (master_dar_final_model_dataset.csv)
and prepares it for machine learning. "Preparing" means:

1. Turning text categories (like "ISIC_Section" = 'G') into numbers,
   because ML models only understand numbers.
2. Creating new helpful numbers ("features") from existing columns,
   e.g. "how many businesses like this already exist in this ward?"
   -- this is our SATURATION signal, used later to avoid recommending
   an already-crowded business type in an area.
3. Building the TARGET column - the thing we want the model to predict
   (Expected Monthly Profit). THIS IS CURRENTLY A PLACEHOLDER - see the
   big warning in get_profit_target() below.

"""

import pandas as pd
import numpy as np

DATA_PATH = './master_dar_final.csv'

SECTOR_NAMES = {
    'A': 'Agriculture, forestry & fishing', 'B': 'Mining & quarrying',
    'C': 'Manufacturing', 'D': 'Electricity & gas',
    'E': 'Water supply & waste management', 'F': 'Construction',
    'G': 'Wholesale & retail trade', 'H': 'Transportation & storage',
    'I': 'Accommodation & food service', 'J': 'Information & communication',
    'K': 'Financial & insurance', 'L': 'Real estate',
    'M': 'Professional & scientific services', 'N': 'Administrative & support services',
    'P': 'Education', 'Q': 'Human health & social work',
    'R': 'Arts, entertainment & recreation', 'S': 'Other service activities',
}

# ------------------------------------------------------------------
# AGE / GENDER SUITABILITY RULES  (PLACEHOLDER - SEE NOTE BELOW)
# ------------------------------------------------------------------
# BEGINNER EXPLANATION: Your SBR data does NOT contain the age or gender
# of business owners, so the model CANNOT learn "which ages/genders do
# well in which sectors" from this data alone.
#
# For now this is a small, simple RULE TABLE based on general domain
# knowledge (e.g. minimum legal working age = 18 in Tanzania). You should
# replace/expand this once your Google Forms survey collects real
# owner age & gender data correlated with sector performance.
#
# min_age / max_age: None means no restriction.
# gender_restriction: None means open to all genders.
AGE_GENDER_RULES = {
    'A': {'min_age': 18, 'max_age': None, 'gender_restriction': None, 'note': 'Agriculture - generally physically demanding'},
    'F': {'min_age': 18, 'max_age': 60, 'gender_restriction': None, 'note': 'Construction - physically demanding'},
    'H': {'min_age': 18, 'max_age': 65, 'gender_restriction': None, 'note': 'Transport - driving licence age limits apply'},
    # Default rule applied to any sector not listed above:
    '_default': {'min_age': 18, 'max_age': None, 'gender_restriction': None, 'note': 'No special restriction'},
}

def get_age_gender_rule(isic_section):
    return AGE_GENDER_RULES.get(isic_section, AGE_GENDER_RULES['_default'])


# ------------------------------------------------------------------
# LOAD + BASIC PREP
# ------------------------------------------------------------------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, low_memory=False)
    return df


def build_features(df):
    """
    Adds engineered columns to the dataframe. Returns the dataframe
    with extra columns - does NOT remove any original columns.

    Safe to call even if these columns already exist in the input file
    (e.g. if you're using master_dar_final_with_profit.csv, which was
    saved AFTER feature engineering) - any pre-existing versions of the
    engineered columns are dropped and recomputed fresh.
    """
    df = df.copy()

    # Drop any pre-existing engineered columns so we don't get
    # _x/_y duplicate-column issues when re-running on an already
    # processed file.
    engineered_cols = ['Census_WorkingAge_Pop', 'Ward_Sector_Count',
                        'District_Sector_Count', 'Log_Capital',
                        'Proxy_Monthly_Profit_TZS', 'Estimated_Monthly_Profit_TZS',
                        'Estimated_Annual_Profit_TZS', 'ROI_Percent_Per_Year',
                        'Breakeven_Months', 'Success_Category']
    df = df.drop(columns=[c for c in engineered_cols if c in df.columns])

    # --- Census "working-age population" as a market-size proxy ---
    # BEGINNER: we add up the 18-60 age columns (male+female) as a rough
    # "how many potential customers/workers are nearby" number.
    working_age_cols = ['Census_18-34_Male', 'Census_18-34_Female',
                         'Census_35-44_Male', 'Census_35-44_Female',
                         'Census_45-60_Male', 'Census_45-60_Female']
    df['Census_WorkingAge_Pop'] = df[working_age_cols].sum(axis=1)

    # --- Saturation signal: how many businesses of this sector already
    # exist in this ward (or district, if ward unknown)? ---
    # BEGINNER: this is the key number that lets us avoid recommending
    # "open a retail shop" in a ward that already has 500 retail shops.
    ward_counts = df.groupby(['Ward_Name', 'ISIC_Section']).size().rename('Ward_Sector_Count')
    df = df.merge(ward_counts, on=['Ward_Name', 'ISIC_Section'], how='left')

    district_counts = df.groupby(['District_Name', 'ISIC_Section']).size().rename('District_Sector_Count')
    df = df.merge(district_counts, on=['District_Name', 'ISIC_Section'], how='left')

    # Log-transform capital (large money values -> easier for linear models)
    df['Log_Capital'] = np.log(df['Estimated_Startup_Capital_TZS'])

    return df


# ------------------------------------------------------------------
# TARGET VARIABLE  --  *** PLACEHOLDER - REPLACE WITH REAL DATA ***
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# TARGET VARIABLE  --  Expected Monthly Profit (TZS)
# ------------------------------------------------------------------
# SOURCES / METHODOLOGY (for your report):
#
# 1. ANCHOR: Tanzania's 2010 National MSME Baseline Survey (FSDT/MIT,
#    Dec 2012) found that "the monthly income from small businesses
#    (TSh 444,200) was significantly higher than the average income of
#    all Tanzanian households (TSh 146,000)."
#
# 2. INFLATION ADJUSTMENT: NBS/IMF data show Tanzania's CPI rose from
#    ~100 (2010 base) to ~120 by late 2025 - but base-year rebasing plus
#    cumulative annual inflation (averaging ~4-6%/yr 2010-2025, e.g. the
#    last 5 years alone were +19.7% cumulative) gives an approximate
#    overall multiplier of ~1.8x from 2010 to 2025/26.
#       TSh 444,200 x 1.8 =~ TSh 800,000 (2025/26 national average
#       monthly MSME profit) -- this is our ANCHOR value.
#
# 3. SECTOR ADJUSTMENT: Sectors differ in typical scale. We reuse the
#    same NBS SBR Table 3.11b (initial capital investment distribution
#    by ISIC section) used for the Capital column: a sector where
#    businesses typically report HIGHER initial capital bands is
#    assumed to also generate proportionally higher profit. We compute
#    a "Sector Capital Index" (weighted average band, 1-5) per section
#    and normalise it into a multiplier centred on 1.0.
#
# 4. OWN-CAPITAL ADJUSTMENT: Within a sector, a business with MORE
#    capital than the sector's typical (median) capital is assumed to
#    generate proportionally more profit, but with diminishing returns
#    (capital_ratio ** 0.4).
#
# 5. RANDOM VARIATION: Real businesses in the same sector with the same
#    capital still earn different amounts (different owners, locations,
#    luck). We add log-normal noise (sigma=0.35, seed=42) to reflect
#    this - this also avoids the model being "too predictable", which
#    is unrealistic.
#
#    Monthly_Profit = ANCHOR x Sector_Multiplier x Capital_Ratio^0.4 x Noise
#
# WHAT TO REPLACE LATER: once TRA/survey data arrives, replace this
# whole function. Keep the output format: one TZS/month number per row.
# ------------------------------------------------------------------

NATIONAL_AVG_MONTHLY_PROFIT_TZS = 800_000  # 2025/26-adjusted anchor (see docstring)

# NBS Table 3.11b - same table used for the Capital column (5 informative
# bands per section: <1M, 1-9.9M, 10-49.9M, 50-99.9M, 100M+), renormalised.
_TABLE_3_11B = {
    'A': [2.6, 19.7, 21.9, 0.7, 2.2], 'B': [1.5, 1.3, 1.8, 1.3, 4.0],
    'C': [2.4, 2.5, 0.8, 0.1, 0.2], 'D': [1.5, 3.5, 2.7, 0.8, 1.2],
    'E': [1.0, 1.0, 1.6, 0.3, 2.3], 'F': [7.8, 28.2, 32.7, 7.8, 15.3],
    'G': [24.9, 51.9, 17.8, 2.2, 1.8], 'H': [6.5, 21.1, 27.6, 8.7, 20.5],
    'I': [20.0, 28.9, 28.6, 10.7, 8.4], 'J': [5.6, 26.3, 27.3, 6.9, 11.9],
    'K': [12.4, 34.0, 12.2, 5.2, 6.7], 'L': [4.5, 9.1, 38.6, 0.0, 29.5],
    'M': [14.2, 36.0, 18.7, 3.1, 3.3], 'N': [15.7, 32.2, 19.9, 5.3, 5.0],
    'P': [1.0, 2.2, 2.4, 1.3, 2.6], 'Q': [3.2, 9.4, 8.1, 2.0, 2.2],
    'R': [18.8, 27.6, 23.0, 3.8, 5.4], 'S': [30.6, 35.3, 8.2, 1.6, 1.3],
}

def _sector_capital_index(section):
    """Weighted-average capital band (1-5) for a sector, from Table 3.11b."""
    probs = np.array(_TABLE_3_11B.get(section, [20, 30, 25, 12, 13]), dtype=float)
    probs = probs / probs.sum()
    return np.sum(probs * np.array([1, 2, 3, 4, 5]))

_SECTOR_INDEX = {sec: _sector_capital_index(sec) for sec in
                 list(_TABLE_3_11B.keys()) + ['T', 'U']}
_MEAN_SECTOR_INDEX = np.mean(list(_SECTOR_INDEX.values()))


def get_profit_target(df, seed=42):
    sector_mult = df['ISIC_Section'].map(_SECTOR_INDEX).fillna(_MEAN_SECTOR_INDEX) / _MEAN_SECTOR_INDEX

    # sector median capital (computed from the data itself)
    sector_median_cap = df.groupby('ISIC_Section')['Estimated_Startup_Capital_TZS'].transform('median')
    capital_ratio = (df['Estimated_Startup_Capital_TZS'] / sector_median_cap).clip(0.1, 10)

    rng = np.random.default_rng(seed)
    noise = np.exp(rng.normal(loc=0, scale=0.35, size=len(df)))

    monthly_profit = (NATIONAL_AVG_MONTHLY_PROFIT_TZS
                       * sector_mult
                       * (capital_ratio ** 0.4)
                       * noise)
    return monthly_profit


if __name__ == '__main__':
    df = load_data()
    df = build_features(df)
    df['Proxy_Monthly_Profit_TZS'] = get_profit_target(df)

    print("Shape after feature engineering:", df.shape)
    print("\nNew columns added:")
    print(['Census_WorkingAge_Pop', 'Ward_Sector_Count', 'District_Sector_Count',
           'Log_Capital', 'Proxy_Monthly_Profit_TZS'])

    print("\nSample (Trade sector, Ilala district):")
    sample = df[(df.ISIC_Section == 'G') & (df.District_Name == 'Ilala')].head(3)
    print(sample[['Ward_Name', 'Estimated_Startup_Capital_TZS', 'Ward_Sector_Count',
                   'Census_WorkingAge_Pop', 'Proxy_Monthly_Profit_TZS']].to_string(index=False))
