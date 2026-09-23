import requests
from bs4 import BeautifulSoup
import csv
import re
import time
from urllib.parse import urljoin

BASE_URL = "https://kysing.kr/popular/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

song_list = []
seen = set()


def parse_page(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [l.strip() for l in text.split("\n") if l.strip() != ""]

    found = 0

    for i, line in enumerate(lines):
        if re.fullmatch(r"\d{4,7}", line):
            song_no = line

            if song_no in seen:
                continue

            rest_lines = []
            for j in range(i + 1, min(i + 8, len(lines))):
                candidate = lines[j]
                if re.fullmatch(r"[▲▼-]\s*\d*", candidate):
                    continue
                if re.fullmatch(r"\d{1,3}", candidate):
                    continue
                if re.fullmatch(r"\d{4}\.\d", candidate):
                    continue
                if re.fullmatch(r"\d{4,7}", candidate):
                    break
                rest_lines.append(candidate)
                if len(rest_lines) >= 2:
                    break

            if len(rest_lines) < 2:
                continue

            title = rest_lines[0]
            singer = rest_lines[1]

            song_list.append({
                "곡번호": song_no,
                "제목": title,
                "가수": singer,
                "브랜드": "금영",
            })
            seen.add(song_no)
            found += 1

    return found, soup


def find_next_page_url(soup):
    links = soup.find_all("a")
    for link in links:
        text = link.get_text(strip=True)
        if "51" in text and "100" in text:
            href = link.get("href")
            if href:
                return urljoin(BASE_URL, href)
    return None


def main():
    song_list.clear()
    seen.clear()

    res = requests.get(BASE_URL, headers=HEADERS, timeout=10)
    res.encoding = "utf-8"
    found, soup = parse_page(res.text)
    print("1페이지 -> " + str(found) + "곡 (누적 " + str(len(song_list)) + ")")

    next_url = find_next_page_url(soup)
    if next_url:
        time.sleep(1)
        res2 = requests.get(next_url, headers=HEADERS, timeout=10)
        res2.encoding = "utf-8"
        found2, soup2 = parse_page(res2.text)
        print("2페이지 -> " + str(found2) + "곡 (누적 " + str(len(song_list)) + ")")

    print("총 " + str(len(song_list)) + "곡 수집됨")

    with open("songs_ky.csv", "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["곡번호", "제목", "가수", "브랜드"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in song_list:
            writer.writerow(s)

    print("songs_ky.csv 저장 완료")


if __name__ == "__main__":
    main()
