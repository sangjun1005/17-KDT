from fastapi import FastAPI, Query
from typing import List, Dict

app = FastAPI()

@app.get('/items/{item_id}')
def read_item(item_id:int):
    return {
        'item_id':item_id
    }

@app.get('/getdata')
def read_items(data:str = 'funcoding'):
    return {
        'data':data
    }

@app.get('/product')
def read_product(q: List[int] = Query([])):
    return{
        'q':q
    }

@app.post('/product-item')
def read_product_item(item: Dict[str,int]):
    return item