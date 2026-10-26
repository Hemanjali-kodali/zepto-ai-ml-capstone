import sqlite3
import pandas as pd


CSV_FILE = "data_pipeline/books_cleaned.csv"
DB_FILE = "data_pipeline/books.db"


df = pd.read_csv(CSV_FILE)

print("Loaded books:", len(df))

connection = sqlite3.connect(DB_FILE)

connection.execute("PRAGMA foreign_keys = ON")

connection.execute("DROP TABLE IF EXISTS books")
connection.execute("DROP TABLE IF EXISTS categories")

connection.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
)
""")

connection.execute("""
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock BOOLEAN,
    availability TEXT,
    category_id INTEGER,
    
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")

categories = df["category"].unique()

for category in categories:

    connection.execute(
        """
        INSERT OR IGNORE INTO categories (category_name)
        VALUES (?)
        """,
        (category,)
    )

category_df = pd.read_sql_query(
    "SELECT * FROM categories",
    connection
)

print("\nCategories table:")
print(category_df)



category_mapping = dict(
    zip(
        category_df["category_name"],
        category_df["category_id"]
    )
)

for _, row in df.iterrows():

    category_id = category_mapping[
        row["category"]
    ]

    connection.execute(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            availability,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            row["in_stock"],
            row["availability"],
            category_id
        )
    )


connection.commit()


book_count = connection.execute(
    "SELECT COUNT(*) FROM books"
).fetchone()[0]

category_count = connection.execute(
    "SELECT COUNT(*) FROM categories"
).fetchone()[0]


print("DATABASE CREATED")

print("Books:", book_count)
print("Categories:", category_count)

print("\nBooks table sample:")

sample = pd.read_sql_query(
    "SELECT * FROM books LIMIT 5",
    connection
)

print(sample)

connection.close()

print("\nDatabase saved to:")
print(DB_FILE)