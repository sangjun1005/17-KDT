from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def read_root():
    return{
        'message':'Hello FastAPI'
    }

@app.get('/items/{item_id}')
def read_item(item_id: int):
    return {
        'item_id':item_id
    }

@app.get('/items')
def read_items(skip: int=0, limit: int = 10):
    return {
        'skip':skip,
        'limit':limit
    }

@app.post('/items/')
def create_item(item:dict):
    return {
        'item':item
    }

@app.put('/items/{item_id}')
def update_item(item_id: int, item: dict):
    return {
        'item_id':item_id,
        'updated_item':item
    }

@app.delete('/items/{item_id}')
def delete_item(item_id:int):
    return{
        'message':f"{item_id} 삭제가 되었습니다."
    }