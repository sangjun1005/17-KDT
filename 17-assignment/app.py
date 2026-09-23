from flask import Flask, render_template, request, send_file
from apscheduler.schedulers.background import BackgroundScheduler
import csv
import io
import crawler_tj
import crawler_ky

app = Flask(__name__)


def load_csv(path):
    songs = []
    try:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                songs.append(row)
    except FileNotFoundError:
        pass
    return songs


def get_songs(keyword, brand, chart):
    if chart == "TJ":
        return load_csv("songs_tj.csv")

    if chart == "금영":
        return load_csv("songs_ky.csv")

    if keyword == "":
        return []

    songs = load_csv("songs_tj.csv") + load_csv("songs_ky.csv")

    if brand != "전체":
        songs = [s for s in songs if s["브랜드"] == brand]

    songs = [s for s in songs if keyword in s["제목"] or keyword in s["가수"]]
    return songs


@app.route("/")
def index():
    keyword = request.args.get("keyword", "").strip()
    brand = request.args.get("brand", "전체")
    chart = request.args.get("chart", "")

    show_results = keyword != "" or chart != ""
    songs = get_songs(keyword, brand, chart)

    return render_template(
        "index.html",
        songs=songs,
        keyword=keyword,
        brand=brand,
        chart=chart,
        show_results=show_results,
        total=len(songs),
        tj_count=len(load_csv("songs_tj.csv")),
        ky_count=len(load_csv("songs_ky.csv")),
    )


@app.route("/download")
def download():
    keyword = request.args.get("keyword", "").strip()
    brand = request.args.get("brand", "전체")
    chart = request.args.get("chart", "")

    songs = get_songs(keyword, brand, chart)

    output = io.StringIO()
    fieldnames = ["곡번호", "제목", "가수", "브랜드"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for s in songs:
        writer.writerow(s)

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8-sig"))
    mem.seek(0)

    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="my_songs.csv")


def refresh_charts():
    print("차트 자동 갱신 시작")
    crawler_tj.main()
    crawler_ky.main()
    print("차트 자동 갱신 끝")


scheduler = BackgroundScheduler()
scheduler.add_job(refresh_charts, "cron", hour=4, minute=0)
scheduler.start()


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
