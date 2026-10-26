import requests

url = "https://books.toscrape.com/"

response = requests.get(url, timeout=10)

print("Status code:", response.status_code)

if response.status_code == 200:
    print("Website connection successful!")
else:
    print("Something went wrong.")