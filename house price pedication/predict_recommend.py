import pandas as pd
import numpy as np
import pickle

# ── Load model ────────────────────────────────────
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('house_data.csv')

FEATURES = [
    'area_sqft', 'bedrooms', 'bathrooms', 'age_years',
    'distance_km', 'floor', 'parking', 'location_score', 'furnished'
]

def recommend_houses(city, budget_lakhs, preferred_bedrooms=None,
                     preferred_area=None, max_distance=None, top_n=5):

    filtered = df[df['city'].str.lower() == city.lower()].copy()

    if filtered.empty:
        print(f"\n  City '{city}' not found in database!")
        print(f"  Available cities: {sorted(df['city'].unique().tolist())}")
        return

    filtered['predicted_price'] = model.predict(filtered[FEATURES])
    filtered = filtered[filtered['predicted_price'] <= budget_lakhs]

    if preferred_bedrooms:
        filtered = filtered[filtered['bedrooms'] >= preferred_bedrooms]
    if preferred_area:
        filtered = filtered[filtered['area_sqft'] >= preferred_area]
    if max_distance:
        filtered = filtered[filtered['distance_km'] <= max_distance]

    if filtered.empty:
        print(f"\n  No houses found in {city} within ₹{budget_lakhs} Lakhs.")
        print("  Try increasing your budget or relaxing filters.")
        return

    filtered = filtered.sort_values(
        ['location_score', 'area_sqft'], ascending=[False, False]
    )

    result = filtered[[
        'city', 'area_sqft', 'bedrooms', 'bathrooms',
        'age_years', 'distance_km', 'location_score',
        'furnished', 'predicted_price'
    ]].head(top_n).reset_index(drop=True)

    result.index += 1  # Start from 1
    result['furnished'] = result['furnished'].map({1: 'Yes', 0: 'No'})
    result['predicted_price'] = result['predicted_price'].round(2)
    result.rename(columns={
        'area_sqft':       'Area (sqft)',
        'bedrooms':        'Beds',
        'bathrooms':       'Baths',
        'age_years':       'Age (yrs)',
        'distance_km':     'Dist (km)',
        'location_score':  'Loc Score',
        'furnished':       'Furnished',
        'predicted_price': 'Price (Lakhs)'
    }, inplace=True)

    print(f"\n  Top {len(result)} Houses in {city} within ₹{budget_lakhs} Lakhs:")
    print("  " + "="*70)
    print(result.to_string())
    print("  " + "="*70)


# ══════════════════════════════════════════════════
#  MAIN — Terminal Input
# ══════════════════════════════════════════════════
print("\n" + "="*50)
print("   🏠 HOUSE RECOMMENDATION SYSTEM")
print("="*50)

available_cities = sorted(df['city'].unique().tolist())
print(f"\n  Available Cities: {', '.join(available_cities)}")

city   = input("\n  Enter City Name   : ").strip()
budget = float(input("  Enter Budget (Lakhs): ").strip())

print("\n  (Press Enter to skip any filter)")
bedrooms  = input("  Min Bedrooms needed : ").strip()
area      = input("  Min Area (sqft)     : ").strip()
distance  = input("  Max Distance (km)   : ").strip()
floor     = input("  Which floor         :").strip()
furnished = input("  Furnished or not(0,1):").strip()

bedrooms = int(bedrooms)     if bedrooms  else None
area     = int(area)         if area      else None
distance = float(distance)   if distance  else None
floor    = int(floor)  if floor else None 
furnished = int(furnished) if furnished else None 

recommend_houses(
    city=city,
    budget_lakhs=budget,
    preferred_bedrooms=bedrooms,
    preferred_area=area,
    max_distance=distance
)
