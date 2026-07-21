import requests
from bs4 import BeautifulSoup   

def fetch_page(url):
    html = requests.get(url, timeout=10).text
    soup = BeautifulSoup(html, "html.parser")
  
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


urls = ["https://", "..."]
for i, url in enumerate(urls):
    with open(f"docs/page_{i}.txt", "w", encoding="utf-8") as f:
        f.write(fetch_page(url))