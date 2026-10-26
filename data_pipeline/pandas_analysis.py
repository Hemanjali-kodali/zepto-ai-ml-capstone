import sqlite3
import pandas as pd


DB_FILE = "data_pipeline/books.db"

connection = sqlite3.connect(DB_FILE)

query_1 = """
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
"""

high_rated_books = pd.read_sql(
    query_1,
    connection
)

print("PANDAS read_sql - QUERY 1")

print(high_rated_books.head(10))


query_2 = """
SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 20 AND 40
"""

books_20_to_40 = pd.read_sql(
    query_2,
    connection
)

print("PANDAS read_sql - QUERY 2")

print(books_20_to_40.head(10))


books_df = pd.read_sql(
    "SELECT * FROM books",
    connection
)

categories_df = pd.read_sql(
    "SELECT * FROM categories",
    connection
)

print("BOOKS DATAFRAME")

print(books_df.head())


print("CATEGORIES DATAFRAME")

print(categories_df)


pandas_join = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

pandas_join = pandas_join[
    [
        "title",
        "price_gbp",
        "rating",
        "category_name"
    ]
]

print("PANDAS MERGE JOIN")

print(pandas_join.head(10))


sql_join_query = """
SELECT
    books.title,
    books.price_gbp,
    books.rating,
    categories.category_name
FROM books
JOIN categories
    ON books.category_id = categories.category_id
"""

sql_join = pd.read_sql(
    sql_join_query,
    connection
)

sql_sorted = sql_join.sort_values(
    by=["title", "price_gbp"]
).reset_index(drop=True)

pandas_sorted = pandas_join.sort_values(
    by=["title", "price_gbp"]
).reset_index(drop=True)


join_equivalent = sql_sorted.equals(
    pandas_sorted
)

print("--------------------------------------------------")
print("JOIN COMPARISON: pd.read_sql vs pd.merge")
print("--------------------------------------------------")

print(f"SQL JOIN row count:    {len(sql_join)}")
print(f"Pandas MERGE row count: {len(pandas_join)}")
print(f"Are SQL JOIN and Pandas MERGE outputs equivalent? {join_equivalent}")

print("\n--- SIDE-BY-SIDE MATCH (First 5 Rows) ---")
print("\n1. pd.read_sql output (from books.db):")
print(sql_sorted.head(5).to_string(index=False))

print("\n2. pd.merge output (in-memory merge):")
print(pandas_sorted.head(5).to_string(index=False))

comparison_preview = pd.DataFrame({
    "Title (SQL)": sql_sorted["title"].head(5),
    "Title (Pandas)": pandas_sorted["title"].head(5),
    "Price GBP (SQL)": sql_sorted["price_gbp"].head(5),
    "Price GBP (Pandas)": pandas_sorted["price_gbp"].head(5),
    "Category (SQL)": sql_sorted["category_name"].head(5),
    "Category (Pandas)": pandas_sorted["category_name"].head(5),
})
print("\n3. Side-by-Side Field Comparison (First 5 Rows):")
print(comparison_preview.to_string(index=False))

connection.close()