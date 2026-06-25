"""
predict.py
===========
BEGINNER EXPLANATION:
This is the "front door" of your system. It has two main functions:

  predict_path_a(...)  -> for someone who HAS a business idea.
                           Tells them: success chance + expected profit
                           + a financial report (ROI, break-even, etc.)
                           Capital is OPTIONAL - if they don't have
                           capital yet, we tell them the typical
                           required capital for that activity instead.

  predict_path_b(...)  -> for someone who has NO idea yet.
                           Returns a ranked list of suggested activities,
                           each with its own financial report.

HOW WE AVOID "EVERYONE GETS THE SAME ANSWER" (anti-saturation):
  1. We count how many businesses of each type ALREADY EXIST in the
     user's ward (Ward_Sector_Count, from saturation_lookup.csv).
     A sector that's already crowded gets a lower score - this naturally
     spreads recommendations out across different activities.
  2. Among the top candidates, we pick using WEIGHTED RANDOM SAMPLING
     (better-scoring activities are more likely to be picked, but it's
     not 100% guaranteed) - so two people with similar inputs can get
     different (but still sensible) suggestions.

HOW AGE/GENDER IS USED:
  Using the rule table in features.py (AGE_GENDER_RULES), sectors that
  don't fit the user's age are filtered out / flagged. Remember: this
  rule table is a placeholder - see features.py for details.
"""

import pandas as pd
import numpy as np
import json
import joblib

from features import SECTOR_NAMES, get_age_gender_rule

MODEL_DIR = './models'

# ------------------------------------------------------------------
# Load all saved artifacts once
# ------------------------------------------------------------------
stage1 = joblib.load(f'{MODEL_DIR}/stage1_linear_regression.joblib')
rf_model = joblib.load(f'{MODEL_DIR}/stage2_random_forest.joblib')
xgb_model = joblib.load(f'{MODEL_DIR}/stage2_xgboost.joblib')

with open(f'{MODEL_DIR}/feature_columns.json') as f:
    cols_info = json.load(f)
STAGE1_COLS = cols_info['stage1_cols']
STAGE2_COLS = cols_info['stage2_cols']
LABEL_MAP = cols_info['label_map']
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

location_lookup = pd.read_csv(f'{MODEL_DIR}/location_lookup.csv')
saturation_lookup = pd.read_csv(f'{MODEL_DIR}/saturation_lookup.csv')
activity_catalog = pd.read_csv(f'{MODEL_DIR}/activity_catalog.csv')

ALL_SECTORS = sorted(activity_catalog['ISIC_Section'].unique())


# ------------------------------------------------------------------
# Financial report - pure formulas (NOT machine learning)
# ------------------------------------------------------------------
def financial_report(capital_tzs, monthly_profit_tzs):
    """
    BEGINNER: These are standard small-business finance formulas:
      - Annual profit  = monthly profit x 12
      - ROI (%)        = (annual profit / capital) x 100
      - Break-even     = how many months until total profit = capital
                          (i.e. capital / monthly profit)
    """
    annual_profit = monthly_profit_tzs * 12
    roi_pct = (annual_profit / capital_tzs) * 100 if capital_tzs > 0 else None
    breakeven_months = (capital_tzs / monthly_profit_tzs) if monthly_profit_tzs > 0 else None

    return {
        'startup_capital_tzs': round(capital_tzs, 0),
        'expected_monthly_profit_tzs': round(monthly_profit_tzs, 0),
        'expected_annual_profit_tzs': round(annual_profit, 0),
        'roi_percent_per_year': round(roi_pct, 1) if roi_pct is not None else None,
        'breakeven_months': round(breakeven_months, 1) if breakeven_months is not None else None,
    }


# ------------------------------------------------------------------
# Build a feature row for one (location, sector, capital) combination
# ------------------------------------------------------------------
def _get_location_info(district, ward=None):
    """Looks up census numbers for a ward, or falls back to district average."""
    if ward is not None:
        row = location_lookup[(location_lookup.District_Name == district) &
                               (location_lookup.Ward_Name == ward)]
        if len(row) > 0:
            return row.iloc[0]
    # fallback: district-level row (Ward_Name is NaN for district-level rows)
    row = location_lookup[(location_lookup.District_Name == district) &
                           (location_lookup.Ward_Name.isna())]
    if len(row) > 0:
        return row.iloc[0]
    raise ValueError(f"District '{district}' not found in location_lookup.")


def _get_saturation(district, ward, isic_section):
    """How many existing businesses of this sector are in this ward/district?"""
    if ward is not None:
        row = saturation_lookup[(saturation_lookup.District_Name == district) &
                                 (saturation_lookup.Ward_Name == ward) &
                                 (saturation_lookup.ISIC_Section == isic_section)]
        if len(row) > 0:
            return row.iloc[0]['Existing_Business_Count']
    row = saturation_lookup[(saturation_lookup.District_Name == district) &
                             (saturation_lookup.Ward_Name.isna()) &
                             (saturation_lookup.ISIC_Section == isic_section)]
    if len(row) > 0:
        return row.iloc[0]['Existing_Business_Count']
    return 0


def _build_feature_row(district, ward, isic_section, capital_tzs, workers=1):
    loc = _get_location_info(district, ward)
    sat_count = _get_saturation(district, ward, isic_section)

    row = {col: 0 for col in STAGE1_COLS}
    row['Log_Capital'] = np.log(max(capital_tzs, 1))
    row['NumberTotalWorkers'] = workers
    row['Census_WorkingAge_Pop'] = loc['Census_WorkingAge_Pop']
    row['Census_Total'] = loc['Census_Total']
    row['Ward_Sector_Count'] = sat_count

    sector_col = f'ISIC_Section_{isic_section}'
    if sector_col in row:
        row[sector_col] = 1

    district_col = f'District_Name_{district}'
    if district_col in row:
        row[district_col] = 1

    return pd.DataFrame([row])[STAGE1_COLS], sat_count


def _predict_for_row(X_row):
    """Runs Stage 1 then Stage 2, returns monthly profit + success category."""
    log_profit_pred = stage1.predict(X_row)[0]
    monthly_profit = np.exp(log_profit_pred)

    X2 = X_row.copy()
    X2['Stage1_Predicted_LogProfit'] = log_profit_pred
    X2 = X2[STAGE2_COLS]

    xgb_pred = xgb_model.predict(X2)[0]
    success_category = INV_LABEL_MAP[xgb_pred]

    return monthly_profit, success_category


# ------------------------------------------------------------------
# PATH A: user HAS a business idea
# ------------------------------------------------------------------
def predict_path_a(isic_detailed, district, ward=None, village=None,
                    capital_tzs=None, age=None, gender=None, workers=1):
    """
    isic_detailed: the 4-digit ISIC code of the activity (from activity_catalog.csv)
    capital_tzs: optional. If None, we use the activity's typical capital.
    """
    activity_rows = activity_catalog[activity_catalog.ISIC_Detailed == isic_detailed]
    if len(activity_rows) == 0:
        raise ValueError(f"Activity ISIC code {isic_detailed} not found in catalog.")
    activity = activity_rows.iloc[0]
    isic_section = activity['ISIC_Section']

    capital_was_provided = capital_tzs is not None
    if capital_tzs is None:
        capital_tzs = activity['Typical_Capital_TZS']

    X_row, sat_count = _build_feature_row(district, ward, isic_section, capital_tzs, workers)
    monthly_profit, success_category = _predict_for_row(X_row)
    report = financial_report(capital_tzs, monthly_profit)

    # Age/gender check
    rule = get_age_gender_rule(isic_section)
    warnings = []
    if age is not None:
        if rule['min_age'] and age < rule['min_age']:
            warnings.append(f"Note: this activity typically requires a minimum age of {rule['min_age']}.")
        if rule['max_age'] and age > rule['max_age']:
            warnings.append(f"Note: this activity may be physically demanding for ages above {rule['max_age']}.")
    if gender is not None and rule['gender_restriction'] is not None:
        if gender.lower() != rule['gender_restriction'].lower():
            warnings.append(f"Note: this activity has historically been associated with {rule['gender_restriction']} owners in this dataset.")

    return {
        'activity': activity['MainActivityDescription'],
        'sector': SECTOR_NAMES.get(isic_section, isic_section),
        'location': {'district': district, 'ward': ward, 'village': village},
        'capital_source': 'user-provided' if capital_was_provided else 'typical for this activity (you did not provide capital)',
        'success_chance': success_category,
        'existing_similar_businesses_in_area': int(sat_count),
        'warnings': warnings,
        **report,
    }


# ------------------------------------------------------------------
# PATH B: user has NO idea -> recommend activities
# ------------------------------------------------------------------
def predict_path_b(district, ward=None, village=None, capital_tzs=None,
                    age=None, gender=None, workers=1, top_n=5,
                    candidate_pool_size=18):
    """
    Returns top_n recommended activities for this person.
    If capital_tzs is None, each candidate uses its own typical capital.
    """
    candidates = []

    # One representative activity per sector (the most common one)
    reps = (activity_catalog.sort_values('Count', ascending=False)
                              .groupby('ISIC_Section').first().reset_index())

    for _, act in reps.iterrows():
        isic_section = act['ISIC_Section']

        # Age/gender filter
        rule = get_age_gender_rule(isic_section)
        if age is not None and rule['min_age'] and age < rule['min_age']:
            continue
        if age is not None and rule['max_age'] and age > rule['max_age']:
            continue
        if gender is not None and rule['gender_restriction'] is not None \
                and gender.lower() != rule['gender_restriction'].lower():
            continue

        cap = capital_tzs if capital_tzs is not None else act['Typical_Capital_TZS']
        X_row, sat_count = _build_feature_row(district, ward, isic_section, cap, workers)
        monthly_profit, success_category = _predict_for_row(X_row)

        # Score = predicted profit, penalised by how saturated the area already is
        saturation_penalty = 1.0 / (1.0 + sat_count)  # more existing businesses -> lower score
        score = monthly_profit * saturation_penalty
        annual_profit = monthly_profit * 12
        roi = (annual_profit / cap) if cap > 0 else 0

        candidates.append({
            'isic_detailed': act['ISIC_Detailed'],
            'activity': act['MainActivityDescription'],
            'sector': SECTOR_NAMES.get(isic_section, isic_section),
            'isic_section': isic_section,
            'capital_tzs': cap,
            'monthly_profit': monthly_profit,
            'roi': roi,
            'existing_similar_businesses_in_area': int(sat_count),
            'score': score,
        })

    cand_df = pd.DataFrame(candidates)

    # ----------------------------------------------------------------
    # BEGINNER: "High/Medium/Low" here is RELATIVE to the other options
    # being shown to THIS person (not the whole dataset). This matters
    # because someone with a lot of capital will see lower ROI% across
    # the board than someone with little capital - without this, almost
    # everything could end up labeled "Low" just because ROI% naturally
    # falls as capital rises. Ranking relative to their own options is
    # more useful: "compared to what else you could do here, this is
    # relatively High/Medium/Low".
    # ----------------------------------------------------------------
    try:
        cand_df['success_category'] = pd.qcut(cand_df['roi'], q=3,
                                                labels=['Low', 'Medium', 'High'],
                                                duplicates='drop')
    except ValueError:
        cand_df['success_category'] = 'Medium'

    cand_df = cand_df.sort_values('score', ascending=False)
    pool = cand_df.head(candidate_pool_size)

    # --- Stratified + weighted random selection ---
    # First, try to guarantee at least one pick from each category present
    # in the pool (so the user sees a spread, not all-Low or all-High).
    # Then fill any remaining slots via weighted random sampling.
    remaining_pool = pool.copy()
    chosen_idx = []

    categories_present = [c for c in ['High', 'Medium', 'Low']
                           if c in remaining_pool['success_category'].values]
    for cat_label in categories_present:
        if len(chosen_idx) >= top_n:
            break
        subset = remaining_pool[remaining_pool['success_category'] == cat_label]
        if len(subset) == 0:
            continue
        weights = subset['score'] / subset['score'].sum()
        pick = np.random.choice(subset.index, size=1, p=weights)[0]
        chosen_idx.append(pick)
        remaining_pool = remaining_pool.drop(pick)

    remaining_needed = top_n - len(chosen_idx)
    if remaining_needed > 0 and len(remaining_pool) > 0:
        weights = remaining_pool['score'] / remaining_pool['score'].sum()
        extra = np.random.choice(remaining_pool.index,
                                  size=min(remaining_needed, len(remaining_pool)),
                                  replace=False, p=weights)
        chosen_idx.extend(extra)

    chosen = cand_df.loc[chosen_idx]

    results = []
    for _, c in chosen.iterrows():
        report = financial_report(c['capital_tzs'], c['monthly_profit'])
        results.append({
            'activity': c['activity'],
            'sector': c['sector'],
            'success_chance': str(c['success_category']),
            'existing_similar_businesses_in_area': c['existing_similar_businesses_in_area'],
            **report,
        })

    return {
        'location': {'district': district, 'ward': ward, 'village': village},
        'recommendations': results,
    }


# ------------------------------------------------------------------
# DEMO
# ------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("DEMO - Path A (has an idea)")
    print("=" * 60)
    sample_activity = activity_catalog.sort_values('Count', ascending=False).iloc[0]
    print(f"Using activity: {sample_activity['MainActivityDescription']} "
          f"(ISIC {sample_activity['ISIC_Detailed']})")
    result_a = predict_path_a(
        isic_detailed=sample_activity['ISIC_Detailed'],
        district='Ilala', ward='Bonyokwa',
        capital_tzs=5_000_000, age=28, gender='female'
    )
    for k, v in result_a.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("DEMO - Path B (no idea yet)")
    print("=" * 60)
    result_b = predict_path_b(district='Ilala', ward='Bonyokwa',
                               capital_tzs=5_000_000, age=28, gender='female', top_n=5)
    print(f"Location: {result_b['location']}")
    for i, r in enumerate(result_b['recommendations'], 1):
        print(f"\n  #{i}: {r['activity']} ({r['sector']})")
        print(f"      Success chance: {r['success_chance']}")
        print(f"      Capital: {r['startup_capital_tzs']:,.0f} TZS")
        print(f"      Expected monthly profit: {r['expected_monthly_profit_tzs']:,.0f} TZS")
        print(f"      ROI/year: {r['roi_percent_per_year']}%  |  Break-even: {r['breakeven_months']} months")
        print(f"      Existing similar businesses nearby: {r['existing_similar_businesses_in_area']}")