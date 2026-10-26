import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin

BASE_URL = "https://books.toscrape.com/"


def get_soup(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def get_categories():
    soup = get_soup(BASE_URL)

    categories = []

    # Find all category links
    category_links = soup.select("div.side_categories ul li ul li a")

    for link in category_links:
        category_name = link.get_text(strip=True)

        category_url = urljoin(
            BASE_URL,
            link.get("href")
        )

        categories.append(
            (category_name, category_url)
        )

    return categories


def scrape_category(category_name, category_url):

    records = []

    current_url = category_url

    while current_url:

        print("Scraping:", category_name, current_url)

        soup = get_soup(current_url)

        books = soup.select("article.product_pod")

        for book in books:

            # Title
            title = book.h3.a["title"]

            # Price
            price = book.select_one(
                ".price_color"
            ).get_text(strip=True)

            # Rating
            rating_element = book.select_one(
                "p.star-rating"
            )

            rating_classes = rating_element.get(
                "class", []
            )

            if len(rating_classes) > 1:
                star_rating = rating_classes[1]
            else:
                star_rating = None

            # Availability
            availability = book.select_one(
                ".availability"
            ).get_text(" ", strip=True)

            records.append({
                "title": title,
                "price": price,
                "star_rating": star_rating,
                "availability": availability,
                "category": category_name
            })

        # Check whether another page exists
        next_link = soup.select_one("li.next a")

        if next_link:
            current_url = urljoin(
                current_url,
                next_link["href"]
            )
        else:
            current_url = None

    return records


def main():

    print("Finding categories...\n")

    categories = get_categories()

    print("Total categories found:",
          len(categories))

    all_books = []

    for category_name, category_url in categories:

        books = scrape_category(
            category_name,
            category_url
        )

        all_books.extend(books)

        unique_categories = set(
            book["category"]
            for book in all_books
        )

        print(
            f"Collected {len(all_books)} books "
            f"from {len(unique_categories)} categories\n"
        )

        # Stop after satisfying assignment requirement
        if (
            len(all_books) >= 60
            and len(unique_categories) >= 3
        ):
            break

    # Convert to DataFrame
    df = pd.DataFrame(all_books)

    # Save raw data
    df.to_csv(
        "data_pipeline/raw_books.csv",
        index=False
    )

    print("\n-----------------------------")
    print("SCRAPING COMPLETED")
    print("-----------------------------")

    print("Total books:", len(df))

    print(
        "Total categories:",
        df["category"].nunique()
    )

    print("\nBooks per category:")
    print(
        df["category"].value_counts()
    )

    print("\nFirst 5 records:")
    print(df.head())


if __name__ == "__main__":
    main()