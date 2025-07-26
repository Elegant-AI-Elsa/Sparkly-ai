import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract all visible text
    texts = soup.stripped_strings
    return '\n'.join(texts)
