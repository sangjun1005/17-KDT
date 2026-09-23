from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def read_root():
    return {
        "번호별 명언 모음집"
    }

@app.get('/1st')
def first():
    return "바쁘다 바빠 현대사회"

@app.get('/2nd')
def second():
    return "이게 팀이야?"

@app.get('/3rd')
def third():
    return "답답하면 니들이 뛰던가"

@app.get('/4th')
def fourth():
    return "사람이 먼저다"

@app.get('/5th')
def fifth():
    return "여러분들 이거 다 거짓말인거 아시죠"

@app.get('/6th')
def sixth():
    return "우리 흥민이 월드클래스 아닙니다"

@app.get('/7th')
def seventh():
    return "야 왜 수영장 ㅋㅋ"

@app.get('/8th')
def eighth():
    return "언~발란스"

@app.get('/9th')
def nineth():
    return "펭현숙 귄카"

@app.get('/10th')
def tenth():
    return "잘한다 잘해"

