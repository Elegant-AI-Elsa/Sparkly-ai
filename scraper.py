import requests
from bs4 import BeautifulSoup

def scrape_website(url, limit_chars=5000):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text[:limit_chars]  # Limit to first N characters
    except Exception as e:
        return f"Error scraping website: {str(e)}"
