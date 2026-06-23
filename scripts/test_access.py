# test_access.py
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.flipkart.com/",
}

url = "https://www.flipkart.com/search?q=men+tshirt&page=1"

response = requests.get(url, headers=HEADERS, timeout=10)
print("Status:", response.status_code)
print("Content length:", len(response.text))
print("First 500 chars:", response.text[:500])