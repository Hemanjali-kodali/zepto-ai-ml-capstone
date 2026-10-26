import pandas as pd

input_file = "data_pipeline/raw_books.csv"

df = pd.read_csv(input_file)

df["title"] = (
    df["title"]
    .str.replace("â", "'", regex=False)
    .str.replace("â€™", "'", regex=False)
    .str.replace("â", "'", regex=False)
)

print("Before cleaning:")
print(df.head())

print("\nData types:")
print(df.dtypes)


df["price_gbp"] = (
    df["price"]
    .str.replace("£", "", regex=False)
    .str.replace("Â", "", regex=False)
    .str.strip()
)

df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)

rating_mapping = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["star_rating"].map(rating_mapping)


df["in_stock"] = (
    df["availability"]
    .str.contains(
        "In stock",
        case=False,
        na=False
    )
)

if df["price_gbp"].isna().any():

    median_price = df["price_gbp"].median()

    df["price_gbp"] = df["price_gbp"].fillna(
        median_price
    )


if df["rating"].isna().any():

    median_rating = round(
        df["rating"].median()
    )

    df["rating"] = df["rating"].fillna(
        median_rating
    )

df["rating"] = df["rating"].astype(int)


GBP_TO_INR = 105.50

df["price_inr"] = (
    df["price_gbp"] * GBP_TO_INR
)


df = df[
    [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "availability",
        "category"
    ]
]

output_file = "data_pipeline/books_cleaned.csv"

df.to_csv(
    output_file,
    index=False
)


print("CLEANING COMPLETED")

print("\nTotal rows:", len(df))

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 cleaned records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nRating values:")
print(df["rating"].unique())

print("\nStock values:")
print(df["in_stock"].unique())

print("\nCleaned file saved to:")
print(output_file)