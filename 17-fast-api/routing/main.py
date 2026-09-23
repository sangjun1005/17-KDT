from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def read_root():
    return {
        'message':'hello fastapi'
    }   

# 경로 매개변수

@app.get('/items/{item_id}')
def read_item(item_id):
    return {
        'item_id':item_id
    }

@app.get('/users/{user_id}/items/{item_name}')
def read_user(user_id, item_name):
    return{
        'user_id':user_id,
        'item_name':item_name
    }

# 쿼리 매개변수

@app.get('/itemname')
def read_name(skip,limit):
    return {
        'skip':skip,
        'limit':limit
    }

@app.get('/products/')
def read_prodducts(skip=0, limit=10):
    return {
        'skip':skip,
        'limit':limit
    }