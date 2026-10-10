# Aspect-based Sentiment Analysis on Hotel Reviews

UE24CS352A Machine Learning mini-project.

Given the text of a TripAdvisor hotel review, the project predicts the **overall star rating** and **five aspect
ratings** (value, rooms, location, cleanliness, sleep quality), then converts stars to polarity
(negative / neutral / positive). It compares three models (Linear SVM, Multinomial Naive Bayes, Ridge regression)
against a majority-class baseline, and includes a live demo function.

## Files
| File | Purpose |
|---|---|
| `absa_hotel_reviews.ipynb` | Main notebook: data loading, EDA, features, models, evaluation, demo, conclusions |
| `requirements.txt` | Python dependencies |
| `outputs/` | Results tables (CSV) and plots produced by the notebook |

## Setup
```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Data
Download **data.csv** (about 361 MB) from the Hugging Face dataset
https://huggingface.co/datasets/jniimi/tripadvisor-review-rating (Files and versions tab) and place it in the
project folder next to the notebook. It is a cleaned, English-only TripAdvisor hotel-review dataset (derived from
Li et al., 2013) with the review text, an overall rating and cleanliness / value / location / rooms / sleep-quality
ratings. The file is not stored in this repo (it is in `.gitignore`).

## Run
```bash
jupyter notebook absa_hotel_reviews.ipynb
```
1. In Step 1 set `DATA_FILE = "data.csv"` and `MAX_REVIEWS` (default 30000; lower it if memory is limited).
2. Click **Kernel -> Restart & Run All** (a few minutes for 30,000 reviews).
3. The last cells run the live demo: edit the text passed to `predict_review(...)` to try any review.

## Outputs (`outputs/`)
`results_all_models.csv`, `baseline_majority.csv`, `overall_vs_aspect.csv`, rating-distribution, model-comparison,
confusion-matrix and aspect-vs-overall plots, and `absa_models.joblib` (saved models, not committed).

## Method
- **Features:** bag of word 1-3 grams, raw counts (no TF-IDF), `min_df=2`, `max_df=20%`, up to 300,000 features.
- **Models (one per aspect):** Linear SVM (class-weight balanced), Multinomial Naive Bayes, Ridge regression (rounded).
- **Split:** 20,000 train / 10,000 test reviews (2/3 - 1/3), fixed random seed.
- **Metrics:** star accuracy, within-1-star accuracy, macro-F1, polarity accuracy; majority-class baseline for context.
- **Extra experiment:** an aspect-trained model vs a model trained only on overall ratings.

## Key findings
See the notebook's Step 12 for details with numbers. In short: Naive Bayes had the best star accuracy; the models clearly
beat the baseline for overall, value, rooms, cleanliness and sleep quality, but not for location (heavily skewed to
5 stars); aspect-specific training helped for cleanliness, sleep quality and location but not value and rooms.

## Limitations
Each model reads the whole review (no sentence-level aspect assignment), ratings are skewed toward 5 stars, and the
dataset has five aspects rather than the seven in the original Wang et al. data.

## Reference
Aspect-based Sentiment Analysis on Hotel Reviews, CS229 Stanford (2016); Li, Ott & Cardie (2013) for the dataset.
