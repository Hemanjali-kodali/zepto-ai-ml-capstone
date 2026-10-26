import sqlite3
import pandas as pd


DB_FILE = "data_pipeline/books.db"
OUTPUT_FILE = "data_pipeline/query_results.txt"

connection = sqlite3.connect(DB_FILE)

queries = {

    "Query 1 - SELECT WHERE": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
    """,

    "Query 2 - ORDER BY": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
    """,

    "Query 3 - LIMIT": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 5
    """,

    "Query 4 - DISTINCT": """
        SELECT DISTINCT category_name
        FROM categories
    """,

    "Query 5 - BETWEEN": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
    """,

    "Query 6 - JOIN": """
        SELECT
            books.title,
            books.price_gbp,
            books.rating,
            categories.category_name
        FROM books
        JOIN categories
            ON books.category_id = categories.category_id
    """
}


with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

    for query_name, query in queries.items():

        print(query_name)

        print(query.strip())

        result = pd.read_sql_query(
            query,
            connection
        )

        print("\nResult:")
        print(result.head(10))

        output.write(query_name + "\n")

        output.write(query.strip())
        output.write("\n\n")

        output.write(
            result.to_string(index=False)
        )

        output.write("\n\n")


connection.close()


print("SQL QUERIES COMPLETED")

print("Total queries:", len(queries))

print("\nResults saved to:")
print(OUTPUT_FILE)