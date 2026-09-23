from flask import Flask, render_template
from scrapper import search_incruit

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/search")
def search():
    jobs = search_incruit()
    return render_template("search.html", jobs=jobs)

if __name__ == "__main__": 
    app.run()