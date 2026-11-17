# Second-Hand Fashion Price Analysis

An analysis of 4,000 second-hand fashion listings in Python, looking at what drives resale price and how quickly items sell.

## Questions

1. How much of the original retail price does a second-hand item keep, by brand tier and condition?
2. Which listings sell, and how fast?
3. Can a simple model predict a fair listing price?

## Data

`data/listings.csv` holds **simulated** listings, generated with `data/generate_data.py`. Each row has the brand, brand tier, category, condition, number of photos, original and listing price, seller country, whether the item sold and the days it took to sell.

Some data quality problems were added on purpose (duplicates, inconsistent brand spelling, missing prices and price typos), so the project includes a real cleaning step.

## What the notebook does

1. **Cleaning:** removes duplicate rows, standardises brand names, fills missing retail prices with the brand and category median, and drops listings priced more than 3x above retail.
2. **Value kept:** compares listing price to retail price by condition, brand tier and brand.
3. **Selling:** sell-through rate and days to sell by condition, number of photos and pricing level.
4. **Price model:** linear regression on log listing price, evaluated on a 20% test set.

## Key findings

- Items new with tags list at about **twice the price** of the same item in good condition.
- Luxury items keep the most value: **77%** of retail when new with tags, against **53%** for high street brands.
- Listings with 8 photos sold **70%** of the time, against **46%** for listings with one photo.
- Listings priced above the usual level for their tier and condition sold **49%** of the time and took a median of **12 days**. Listings priced below that level sold **66%** of the time in **7 days**.
- The regression explains **94%** of the variation in log price, with a mean absolute error of about **€9**.

Because the data is simulated, these results demonstrate the method rather than real market behaviour.

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook resale_price_analysis.ipynb
```

To regenerate the data: `python data/generate_data.py`

## Tools

Python, pandas, NumPy, Matplotlib, scikit-learn, Jupyter
