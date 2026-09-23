import requests
import csv
import time
from datetime import date, timedelta

API_URL = "https://www.tjmedia.com/legacy/api/topAndHot100"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://www.tjmedia.com/chart/top100",
}

song_list = []
seen = set()


def get_date_windows(months_back=24):
    windows = []
    today = date.today()
    cursor = today
    for i in range(months_back * 2):
        end = cursor
        start = end - timedelta(days=14)
        windows.append((start.isoformat(), end.isoformat()))
        cursor = start - timedelta(days=1)
    return windows


def fetch_chart(start_date, end_date):
    payload = {
        "chartType": "TOP",
        "searchStartDate": start_date,
        "searchEndDate": end_date,
        "strType": "",
    }
    try:
        res = requests.post(API_URL, data=payload, headers=HEADERS, timeout=10)
        data = res.json()
    except Exception as e:
        print("요청실패", e)
        return []

    results = []

    def walk(obj):
        if isinstance(obj, dict):
            if "rank" in obj and "pro" in obj:
                results.append(obj)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return results


def main():
    song_list.clear()
    seen.clear()

    windows = get_date_windows(months_back=24)

    for start, end in windows:
        rows = fetch_chart(start, end)
        new_count = 0
        for row in rows:
            song_no = str(row.get("pro", "")).strip()
            if song_no == "" or song_no in seen:
                continue
            song_list.append({
                "곡번호": song_no,
                "제목": row.get("indexTitle", ""),
                "가수": row.get("indexSong", ""),
                "브랜드": "TJ",
            })
            seen.add(song_no)
            new_count += 1
        print(start + " ~ " + end + " 새곡 " + str(new_count) + "개 누적 " + str(len(song_list)))
        time.sleep(0.5)
        if len(song_list) >= 500:
            break

    print("총 " + str(len(song_list)) + "곡")

    with open("songs_tj.csv", "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["곡번호", "제목", "가수", "브랜드"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in song_list:
            writer.writerow(s)

    print("songs_tj.csv 저장 완료")


if __name__ == "__main__":
    main()
