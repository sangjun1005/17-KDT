from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse,HTMLResponse,PlainTextResponse,RedirectResponse

app = FastAPI()

@app.get('/redirect')
def read_redirect():
    return RedirectResponse(url='/html')


@app.get('/html',response_class=HTMLResponse)
def read_html():
    return '<h1>This is HTML</h1>'

@app.get('/json',response_class=JSONResponse)
def read_json():
    return {
        'msg':'This is JSON'
    }

@app.get('/text',response_class=PlainTextResponse)
def read_text():
    return 'This is Plain Text'

# class Item(BaseModel):
#     name:str
#     description:str=None
#     prices:float

# def get_item_from_db(id):
#     return {
#         'name':'simple item',
#         'description':'a simple item description',
#         'prices':50.0,
        
#     }

# @app.get('/items/{item_id}',response_model=Item)
# def read_item(item_id:int):
#     items = get_item_from_db(item_id)
#     return items