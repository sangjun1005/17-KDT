from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Union

app = FastAPI()

# class Item(BaseModel):
#     name: str
#     price: float
#     is_offer: bool = None

# @app.post('/items/')
# def create_item(item: Item):
#     return {
#         'item':item
#     }

# # 필드 제약조건

# class Item(BaseModel):
#     name: str = Field(..., title='Item Name', min_length=2, max_length=50)
#     description: str = Field(None, description='The Description of the item', max_length=300)
#     price: float = Field(..., gt=0, description='0보다 커야한다.')

# @app.post('/items/')
# def create_item(items: Item):
#     return {
#         'item': items
#     }

class Item(BaseModel):
    name:str
    tags:List[str]
    variant: Union[int, str]


@app.post('items/')
def create_item(item: Item):
    return {
        'item':item
    }