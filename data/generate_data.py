"""
Generates the synthetic dataset used in this project (data/listings.csv).

The data is simulated, not real marketplace data. It is built to behave like
second-hand fashion listings: brand tier, item condition and photo count affect
the asking price and how fast an item sells. A few data quality problems
(missing values, duplicate rows, inconsistent brand spelling, typos in prices)
are added on purpose so the notebook has real cleaning work to do.

Run:  python data/generate_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 4000

brands = {
    # brand: (tier, typical original retail price in EUR)
    "Zara": ("high street", 40), "H&M": ("high street", 25), "Mango": ("high street", 45),
    "Pull&Bear": ("high street", 30), "Bershka": ("high street", 25), "Reserved": ("high street", 30),
    "Nike": ("sportswear", 80), "Adidas": ("sportswear", 75), "Puma": ("sportswear", 60),
    "The North Face": ("premium", 150), "Levi's": ("premium", 90), "Tommy Hilfiger": ("premium", 110),
    "Ralph Lauren": ("premium", 130), "COS": ("premium", 90),
    "Gucci": ("luxury", 700), "Prada": ("luxury", 800), "Burberry": ("luxury", 650),
}
categories = {
    # category: price multiplier vs brand typical price
    "T-shirts": 0.5, "Jeans": 1.0, "Dresses": 1.1, "Jumpers": 1.0, "Jackets": 2.0,
    "Coats": 2.6, "Trainers": 1.3, "Bags": 1.8,
}
conditions = ["New with tags", "New without tags", "Very good", "Good", "Satisfactory"]
cond_weights = [0.10, 0.12, 0.40, 0.30, 0.08]
cond_keep = {"New with tags": 0.62, "New without tags": 0.52, "Very good": 0.40,
             "Good": 0.30, "Satisfactory": 0.20}  # share of retail price kept
tier_keep = {"high street": 0.85, "sportswear": 1.0, "premium": 1.1, "luxury": 1.35}
countries = ["Lithuania", "Poland", "France", "Germany", "Spain", "Italy"]
country_w = [0.15, 0.20, 0.22, 0.18, 0.13, 0.12]

brand_names = list(brands)
brand_w = np.array([9, 10, 7, 6, 6, 6, 8, 7, 4, 4, 5, 4, 3, 3, 1, 0.8, 1.2])
brand_w = brand_w / brand_w.sum()

rows = []
for i in range(N):
    brand = rng.choice(brand_names, p=brand_w)
    tier, base = brands[brand]
    cat = rng.choice(list(categories))
    cond = rng.choice(conditions, p=cond_weights)
    photos = int(rng.integers(1, 9))
    original = round(base * categories[cat] * rng.uniform(0.8, 1.25), 2)
    keep = cond_keep[cond] * tier_keep[tier] * rng.normal(1, 0.25)
    listing_price = max(2.0, round(original * keep, 0))
    listed = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))

    # Chance of selling: better condition, more photos and fair pricing help.
    fair_ratio = listing_price / (original * cond_keep[cond] * tier_keep[tier])
    p_sell = 0.55 + 0.04 * (photos - 4) - 0.35 * (fair_ratio - 1) + (0.08 if cond.startswith("New") else 0)
    sold = rng.random() < np.clip(p_sell, 0.05, 0.95)
    days = None
    if sold:
        days = int(np.clip(rng.gamma(2.0, 6 + 8 * max(fair_ratio - 0.9, 0)) - photos * 0.6, 1, 120))
    rows.append({
        "listing_id": 100000 + i, "brand": brand, "brand_tier": tier, "category": cat,
        "condition": cond, "photos": photos, "original_price_eur": original,
        "listing_price_eur": listing_price, "listed_date": listed.date().isoformat(),
        "seller_country": rng.choice(countries, p=country_w), "sold": sold, "days_to_sell": days,
    })

df = pd.DataFrame(rows)

# --- Add realistic data quality issues on purpose ---
idx = rng.choice(df.index, 60, replace=False)
df.loc[idx, "brand"] = df.loc[idx, "brand"].str.upper()           # inconsistent spelling
idx = rng.choice(df.index, 25, replace=False)
df.loc[idx, "brand"] = df.loc[idx, "brand"].str.lower() + " "       # lower case + trailing space
idx = rng.choice(df.index, 40, replace=False)
df.loc[idx, "original_price_eur"] = np.nan                          # missing values
idx = rng.choice(df.index, 6, replace=False)
df.loc[idx, "listing_price_eur"] = df.loc[idx, "listing_price_eur"] * 100  # price typos
df = pd.concat([df, df.sample(30, random_state=1)])                 # duplicate rows
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("data/listings.csv", index=False)
print(f"Saved {len(df)} rows to data/listings.csv")
