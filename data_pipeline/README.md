# Module 1 — Data Pipeline: Catalog Intelligence & Relational ETL

This module implements an end-to-end data engineering pipeline designed to benchmark catalog-style pricing and availability data before reaching downstream analytics dashboards.

## Pipeline Architecture & Mechanics

```
Web Scraping (books.toscrape.com)
             │
             ▼
   raw_books.csv (69 items across 3 categories)
             │
             ▼
   cleaning.py (imputation, encoding fixes, currency conversion to INR)
             │
             ▼
   books_cleaned.csv
             │
             ▼
   database.py ──► SQLite (books.db with normalized 2-table PK/FK schema)
             │
     ┌───────┴────────┐
     ▼                ▼
queries.py      pandas_analysis.py
(6 SQL queries)   (pd.read_sql vs pd.merge equivalence)
```

## Data Source
- **Website**: `https://books.toscrape.com/` (public scraping practice site, keyless, no login/rate limits).
- **Scope**: Scrapes all books across 3 categories:
  - **Travel**: 11 books
  - **Mystery**: 32 books
  - **Historical Fiction**: 26 books
  - **Total**: 69 books (exceeds the ≥ 60 books requirement).
- **Scraped Fields**: `title`, `price` (as listed, in GBP), `star_rating` (as text), `availability` (as text), and `category`.

## Parsing & Cleaning Decisions
Implemented in `data_pipeline/cleaning.py`:
1. **Price (`price_gbp`)**: Currency symbols (`£`, `Â`) stripped and cast to `float64`.
2. **Rating (`rating`)**: Star rating word strings (`"One"`–`"Five"`) mapped to integer scale `1–5` (`int64`).
3. **Availability (`in_stock`)**: Parsed to a boolean column (`bool`), `True` if containing `"In stock"`.
4. **Encoding Artifacts**: Resolved UTF-8 mojibake (e.g. `â\x80\x99` cleaned to `'`).
5. **Missing Value Strategy & Justification**:
   - Any missing numeric values are imputed using **median imputation** (and rounded median for integer rating) rather than dropping rows.
   - *Justification*: In catalog benchmarking and retail intelligence, dropping rows artificially shrinks catalog size and category assortment metrics. Median imputation preserves complete category representation and catalog breadth while preventing price outliers from distorting metrics.

## Currency Conversion Baseline
- **Required Fixed-Rate Baseline**: **`1 GBP = 105.50 INR`**
- Computed as `price_inr = price_gbp * 105.50`.
- This is a project-defined constant with no date reference or external API dependency.

## Normalized Relational Database Schema
Implemented in `data_pipeline/database.py` targeting `data_pipeline/books.db`:

```sql
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock BOOLEAN,
    availability TEXT,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
```
*Note: `database.py` drops and regenerates tables cleanly, ensuring idempotent execution.*

## SQL Queries & Pandas Equivalence
1. **SQL Queries (`queries.py`)**:
   - Executes 6 queries covering:
     - `SELECT / WHERE`: Filter books by rating (`rating >= 4`)
     - `ORDER BY`: Sort books by price descending (`ORDER BY price_gbp DESC`)
     - `LIMIT`: Top 5 most expensive books
     - `DISTINCT`: Unique category listing
     - `BETWEEN`: Price range filtering (`price_gbp BETWEEN 20 AND 40`)
     - `JOIN`: Relational inner join between `books` and `categories`
   - Results are printed to console and saved in `data_pipeline/query_results.txt`.
2. **Pandas Equivalence (`pandas_analysis.py`)**:
   - Queries `books.db` using `pd.read_sql` for filtered queries.
   - Reads `books` and `categories` tables into memory and performs `pd.merge(...)`.
   - Compares the `pd.read_sql` JOIN output and `pd.merge` output side-by-side and verifies equality (`join_equivalent = True`).

## Installation & Execution Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test connectivity
python data_pipeline/test_connection.py

# 3. Run scraper
python data_pipeline/scraper.py

# 4. Clean data and apply currency conversion
python data_pipeline/cleaning.py

# 5. Populate SQLite database
python data_pipeline/database.py

# 6. Execute SQL queries
python data_pipeline/queries.py

# 7. Verify SQL vs. Pandas join equivalence
python data_pipeline/pandas_analysis.py
```
