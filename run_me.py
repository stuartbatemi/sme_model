"""
run_me.py
=========
JUST RUN THIS FILE. It will ASK YOU QUESTIONS in the terminal -
you don't need to open or edit any code.

      python3 run_me.py

You need these files in the SAME FOLDER as this one:
  - master_dar_final.csv
  - features.py
  - predict.py
  - the 'models' folder (created by running train_model.py once, first time only)

If the 'models' folder doesn't exist yet, run this ONE TIME first:
      python3 train_model.py
"""

import re
from predict import predict_path_a, predict_path_b, activity_catalog


VALID_DISTRICTS = ['Ilala', 'Kigamboni', 'Kinondoni', 'Temeke', 'Ubungo']


# ------------------------------------------------------------------
# Simple English <-> Swahili keyword dictionary, so people who don't
# know ISIC codes (or Swahili) can still find a matching activity.
# Add more words here any time - just follow the same pattern.
# ------------------------------------------------------------------
KEYWORD_MAP = {
    'clothes': ['NGUO'], 'clothing': ['NGUO'], 'fashion': ['NGUO'],
    'food': ['CHAKULA', 'VYAKULA'], 'restaurant': ['CHAKULA', 'MGAHAWA'],
    'drinks': ['VINYWAJI'], 'beverage': ['VINYWAJI'], 'juice': ['VINYWAJI'],
    'shop': ['REJAREJA', 'DUKA'], 'retail': ['REJAREJA'], 'store': ['DUKA'],
    'wholesale': ['JUMLA'],
    'construction': ['UJENZI'], 'building': ['UJENZI'],
    'electronics': ['UMEME'], 'electrical': ['UMEME'],
    'medicine': ['DAWA'], 'pharmacy': ['DAWA'], 'drugs': ['DAWA'],
    'grain': ['NAFAKA'], 'cereal': ['NAFAKA'], 'maize': ['NAFAKA'],
    'spare': ['SPARE'], 'parts': ['SPARE'],
    'car': ['MAGARI'], 'vehicle': ['MAGARI'], 'cars': ['MAGARI'],
    'motorcycle': ['PIKIPIKI'], 'bike': ['PIKIPIKI'], 'bajaji': ['PIKIPIKI'],
    'phone': ['SIMU'], 'mobile': ['SIMU'],
    'cosmetics': ['VIPODOZI', 'UREMBO'], 'beauty': ['UREMBO', 'VIPODOZI'],
    'shoes': ['VIATU'], 'footwear': ['VIATU'],
    'children': ['WATOTO'], 'kids': ['WATOTO'], 'baby': ['WATOTO'],
    'education': ['ELIMU'], 'school': ['ELIMU'], 'tuition': ['ELIMU'],
    'health': ['AFYA'], 'clinic': ['AFYA'], 'medical': ['AFYA'],
    'transport': ['USAFIRISHAJI'], 'delivery': ['USAFIRISHAJI'],
    'bakery': ['MIKATE'], 'bread': ['MIKATE'],
    'hair': ['NYWELE'], 'salon': ['NYWELE', 'SALONI'], 'barber': ['NYWELE'],
    'household': ['MAJUMBANI', 'NYUMBANI'], 'furniture': ['SAMANI'],
    'welding': ['UCHOMELEAJI'], 'carpentry': ['USELEMALA'],
    'tailoring': ['USHONAJI'], 'tailor': ['USHONAJI'], 'sewing': ['USHONAJI'],
    'printing': ['UCHAPISHAJI'],
    'bar': ['BARI'], 'pub': ['BARI'],
    'repair': ['UFUNDI'], 'garage': ['UFUNDI', 'GARAGE'],
    'fish': ['SAMAKI'], 'farming': ['KILIMO'], 'agriculture': ['KILIMO'],
    'computer': ['KOMPYUTA'], 'internet': ['KOMPYUTA'],
}


def search_activities(query):
    """
    Searches activity_catalog for entries matching the user's words.
    Works with English or Swahili words.
    Returns a ranked list of candidate rows.
    """
    words = re.findall(r'[A-Za-z]+', query.upper())
    search_terms = set()
    for w in words:
        search_terms.add(w)  # keep original word too (covers Swahili input)
        for eng, swahili_list in KEYWORD_MAP.items():
            if eng.upper() == w or eng.upper() in w or w in eng.upper():
                search_terms.update(swahili_list)

    if not search_terms:
        return activity_catalog.head(0)

    def score(desc):
        desc_upper = str(desc).upper()
        return sum(1 for t in search_terms if t in desc_upper)

    cat = activity_catalog.copy()
    cat['_score'] = cat['MainActivityDescription'].apply(score)
    cat = cat[cat['_score'] > 0]
    cat = cat.sort_values(['_score', 'Count'], ascending=[False, False])
    return cat.head(8)


# ------------------------------------------------------------------
# Helper input functions
# ------------------------------------------------------------------
def ask_yes_no(prompt):
    while True:
        ans = input(prompt + " (yes/no): ").strip().lower()
        if ans in ('yes', 'y'):
            return True
        if ans in ('no', 'n'):
            return False
        print("  Please type 'yes' or 'no'.")


def ask_district():
    while True:
        print(f"\nValid districts: {', '.join(VALID_DISTRICTS)}")
        d = input("Which district? ").strip().title()
        if d in VALID_DISTRICTS:
            return d
        print("  Sorry, that district isn't currently accounted for yet. Please pick any from the list above.")


def ask_ward():
    w = input("Which ward do you think of starting the business in? (or press Enter if you don't know): ").strip()
    return w if w else None


def ask_village():
    v = input("Which village/area? (optional, press Enter to skip): ").strip()
    return v if v else None


def ask_capital():
    while True:
        c = input("How much starting capital do you have, in TZS? "
                   "(or press Enter if you have none yet): ").strip()
        if c == '':
            return None
        c = c.replace(',', '')
        try:
            return float(c)
        except ValueError:
            print("  Please enter a number (e.g. 5000000), or press Enter for none.")


def ask_age():
    while True:
        a = input("What is your age? ").strip()
        try:
            return int(a)
        except ValueError:
            print("  Please enter a number.")


def ask_gender():
    while True:
        g = input("What is your gender? (male/female): ").strip().lower()
        if g in ('male', 'female'):
            return g
        print("  Please type 'male' or 'female'.")


def print_line():
    print("-" * 55)


# ------------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------------
print_line()
print("SME BUSINESS ADVISOR - Dar es Salaam")
print_line()

name = input("\nWhat is your name? ").strip()
if not name:
    name = "there"

print(f"\nHello {name}, welcome to SME Advisor!")

has_idea = ask_yes_no("Do you have a business idea?")

print()
district = ask_district()
ward = ask_ward()
village = ask_village()
capital = ask_capital()
age = ask_age()
gender = ask_gender()

if has_idea:
    chosen = None
    while chosen is None:
        query = input("\nDescribe your business idea in a few words "
                       "(English or Swahili, e.g. 'selling clothes', "
                       "'bakery', 'phone repair'): ").strip()
        matches = search_activities(query)

        if len(matches) == 0:
            print("  No matches found. Try different/simpler words "
                  "(e.g. just 'clothes' or 'food').")
            continue

        print("\nI found these matching activities:")
        for i, (_, row) in enumerate(matches.iterrows(), 1):
            print(f"  {i}. {row['MainActivityDescription']} "
                  f"({row['Sector_Name']})")
        print(f"  0. None of these - search again")

        pick = input("\nType the number of the best match: ").strip()
        if pick == '0':
            continue
        try:
            idx = int(pick) - 1
            if 0 <= idx < len(matches):
                chosen = matches.iloc[idx]
            else:
                print("  Please type a valid number from the list.")
        except ValueError:
            print("  Please type a valid number from the list.")

    print()
    print_line()
    print("RESULT")
    print_line()
    print(f"Activity: {chosen['MainActivityDescription']}")
    print(f"Sector:   {chosen['Sector_Name']}")

    result = predict_path_a(
        isic_detailed=chosen['ISIC_Detailed'],
        district=district, ward=ward, village=village,
        capital_tzs=capital, age=age, gender=gender,
    )

    print()
    print(f"Capital used: {result['startup_capital_tzs']:,.0f} TZS  "
          f"({result['capital_source']})")
    print()
    print(f">>> SUCCESS CHANCE: {result['success_chance']}")
    print()
    print(f"Expected monthly profit:  {result['expected_monthly_profit_tzs']:,.0f} TZS")
    print(f"Expected annual profit:   {result['expected_annual_profit_tzs']:,.0f} TZS")
    print(f"ROI per year:             {result['roi_percent_per_year']}%")
    print(f"Break-even time:          {result['breakeven_months']} months")
    print()
    print(f"Existing similar businesses nearby: {result['existing_similar_businesses_in_area']}")

    if result['warnings']:
        print("\nNotes / warnings:")
        for w in result['warnings']:
            print(f"  - {w}")

else:
    how_many = input("\nHow many suggestions would you like? (default 5): ").strip()
    how_many = int(how_many) if how_many.isdigit() else 5

    result = predict_path_b(
        district=district, ward=ward, village=village,
        capital_tzs=capital, age=age, gender=gender, top_n=how_many,
    )

    print()
    print_line()
    print("SUGGESTED BUSINESS IDEAS FOR YOU")
    print_line()

    for i, rec in enumerate(result['recommendations'], 1):
        print_line()
        print(f"#{i}: {rec['activity']}")
        print(f"Sector: {rec['sector']}")
        print(f"Success chance: {rec['success_chance']}")
        print()
        print(f"  Required/used capital:   {rec['startup_capital_tzs']:,.0f} TZS")
        print(f"  Expected monthly profit: {rec['expected_monthly_profit_tzs']:,.0f} TZS")
        print(f"  Expected annual profit:  {rec['expected_annual_profit_tzs']:,.0f} TZS")
        print(f"  ROI per year:            {rec['roi_percent_per_year']}%")
        print(f"  Break-even time:         {rec['breakeven_months']} months")
        print(f"  Existing similar businesses nearby: {rec['existing_similar_businesses_in_area']}")

print()
print_line()
print("Welcome thank you for using SME advisor!.\n NOTE: ALL RESPONSES ARE BASED ON HISTORICAL DATA AND PREDICTIVE MODELS, AND SHOULD BE TAKEN AS ROUGH ESTIMATES RATHER THAN GUARANTEES. ")