import requests
from bs4 import BeautifulSoup   

def fetch_page(url):
    html = requests.get(url, timeout=10).text
    soup = BeautifulSoup(html, "html.parser")
    # strip scripts/styles, keep visible text
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # collapse blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

# save each page to docs/
urls = ["https://", "..."]
for i, url in enumerate(urls):
    with open(f"docs/page_{i}.txt", "w", encoding="utf-8") as f:
        f.write(fetch_page(url))