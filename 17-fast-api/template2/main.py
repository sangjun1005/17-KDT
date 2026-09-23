from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

app = FastAPI()

# html 렌더링 디렉토리
templates = Jinja2Templates(directory='templates')

@app.get('/')
def read_root(request: Request):
    print(request)
    return templates.TemplateResponse(request, "index.html")