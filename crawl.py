import requests
from bs4 import BeautifulSoup
from os.path import dirname, join, abspath
from datetime import datetime
import sys

sys.stdout.reconfigure(encoding='utf-8')  # Cho phép in Unicode

def extract_filename(url):
    last_slash_index = url.rindex("/")
    last_period_index = url.rindex(".")
    last_slash = last_slash_index + 1
    output = url[last_slash:last_period_index]
    return output

def crawl_website(url, filepath):
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{date_string} Crawl {url}")

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    parent_element = soup.find(class_="cldivContentDocVn")
    child_element = parent_element.find(class_="content1")
    paragraphs = child_element.find_all("p")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("")

    with open(filepath, "a", encoding="utf-8") as f:
        for p in paragraphs:
            text = p.get_text() + "\n"
            text = text.replace("\r\n", " ")
            f.write(text)

CWD = abspath(dirname(__file__))
OUTPUT_FOLDER = join(CWD, "data", "raw")
print(OUTPUT_FOLDER)

with open(join(CWD, "error.log"), "w", encoding="utf-8") as f:
    f.write("")

log_file = open(join(CWD, "error.log"), "a", encoding="utf-8")
urls = [
    "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Luat-Phong-chong-ma-tuy-2021-445185.aspx"
]

for i, url in enumerate(urls):
    filepath = extract_filename(url)
    try:
        crawl_website(url, join(OUTPUT_FOLDER, f"{filepath}.txt"))
    except Exception as e:
        print(e)
        log_file.write(url + "\n")
