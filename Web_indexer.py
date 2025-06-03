from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urljoin

visited = set()
links = []

def dis_link_valid(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def get_links(url):
    if not dis_link_valid(url):
        return

    # Retry until request succeeds
    while True:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            print('Retrying due to error:', e)
            continue

    page = BeautifulSoup(response.text, 'html.parser')
    base_url = response.url

    for link in page.find_all('a'):
        href = link.get('href')
        if href:
            abs_url = urljoin(base_url, href)
            if dis_link_valid(abs_url) and abs_url not in visited:
                visited.add(abs_url)
                links.append(abs_url)
                yield abs_url

def index(url):
    if not dis_link_valid(url):
        return ""
    try:
        response = requests.get(url, timeout=(10, 60))
        response.raise_for_status()
        page = BeautifulSoup(response.text, 'html.parser')
        for script in page.find_all('script'):
            script.decompose()
        return page.prettify()
    except Exception as e:
        print('Error indexing:', e)
        return ""

# Ensure the URL has a valid scheme
url = "https://www.delish.com/"
for href in get_links(url):
    cleaned_text = index(href)
    print(f"Indexed: {href}")
    print(f"Text: {cleaned_text}")
