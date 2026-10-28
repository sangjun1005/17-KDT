from pathlib import Path
import math
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium

OUT = Path(__file__).parent
pdfmetrics.registerFont(TTFont('Korean', 'C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('KoreanBold', 'C:/Windows/Fonts/malgunbd.ttf'))
W, H = A4
c = canvas.Canvas(str(OUT / 'doorlock-flowchart.pdf'), pagesize=A4)
c.setTitle('도어락 동작 플로우차트')
INK = HexColor('#25344B')
BLUE = HexColor('#EAF2FC')
GREEN = HexColor('#EAF6F0')
RED = HexColor('#FFF0EF')

def text(x, y, s, size=10.5, bold=False):
    c.setFillColor(INK)
    c.setFont('KoreanBold' if bold else 'Korean', size)
    lines = s.split('\n')
    for i, line in enumerate(lines):
        c.drawCentredString(x, H-y+(len(lines)-1)*7-i*14-4, line)

def box(x, y, s, w=202, h=44, fill=BLUE, pill=False):
    c.setStrokeColor(INK)
    c.setLineWidth(.85)
    c.setFillColor(fill)
    c.roundRect(x-w/2, H-y-h/2, w, h, h/2 if pill else 5, fill=1, stroke=1)
    text(x, y, s)

def diamond(x, y, s, w=180, h=64):
    p = c.beginPath()
    p.moveTo(x, H-y+h/2)
    p.lineTo(x+w/2, H-y)
    p.lineTo(x, H-y-h/2)
    p.lineTo(x-w/2, H-y)
    p.close()
    c.setFillColor(HexColor('#FFF6DF'))
    c.setStrokeColor(INK)
    c.drawPath(p, fill=1, stroke=1)
    text(x,y,s)

def arrow(points, label=None, label_at=None):
    c.setStrokeColor(INK)
    c.setLineWidth(.9)
    p = c.beginPath()
    p.moveTo(points[0][0], H-points[0][1])
    for x,y in points[1:]: p.lineTo(x,H-y)
    c.drawPath(p)
    x,y = points[-1]
    px,py = points[-2]
    a = math.atan2(y-py,x-px)
    tip = c.beginPath()
    tip.moveTo(x,H-y)
    for d in (-.45,.45):
        tip.lineTo(x-7*math.cos(a+d), H-(y-7*math.sin(a+d)))
        tip.moveTo(x,H-y)
    c.drawPath(tip)
    if label: text(*label_at,label,9)

def header(n,title,subtitle):
    text(W/2,42,title,19,True)
    text(W/2,70,subtitle,10)
    c.setStrokeColor(HexColor('#CFD7E2'))
    c.line(30,H-88,W-30,H-88)
    text(W/2,814,f'Wokwi · Raspberry Pi Pico · MicroPython     |     {n} / 2',9)

header(1,'도어락 동작 플로우차트','01  비밀번호 입력 · 문 열림 · 5초 후 자동 잠금')
X=330
box(X,118,'시작',w=130,h=30,pill=True)
box(X,174,'문 닫기 · 빨간 LED 켜기\n초기 비밀번호 3152')
box(X,238,'키패드 입력 대기\n숫자 최대 4자리 입력 후 #')
diamond(X,318,'4자리 입력 완료?')
diamond(X,414,'비밀번호 일치?')
box(98,318,'4자리 입력 안내',w=140,h=40,fill=RED)
box(98,414,'오류 안내\n입력 내용 초기화',w=140,h=44,fill=RED)
box(X,496,'초록 LED 켜기\n서보모터 0° → 90°',fill=GREEN)
box(X,558,'문 완전히 열림\n5초 타이머 시작',fill=GREEN)
diamond(X,634,'5초 경과 또는\nD 입력?',h=70)
box(X,712,'서보모터 90° → 0°\n문 닫기 · 빨간 LED 켜기')
arrow([(X,133),(X,152)])
arrow([(X,196),(X,216)])
arrow([(X,260),(X,286)])
arrow([(X,350),(X,382)],'예',(345,366))
arrow([(240,318),(168,318)],'아니요',(204,305))
arrow([(98,298),(98,238),(229,238)])
arrow([(240,414),(168,414)],'아니요',(204,401))
arrow([(98,436),(98,465),(22,465),(22,210),(330,210),(330,216)])
arrow([(X,446),(X,474)],'예',(345,460))
arrow([(X,518),(X,536)])
arrow([(X,580),(X,599)])
arrow([(420,634),(480,634),(480,590),(330,590),(330,599)],'아니요',(455,619))
arrow([(X,669),(X,690)],'예',(345,680))
arrow([(X,734),(540,734),(540,238),(431,238)])
text(295,765,'* 한 글자 삭제  |  B 전체 삭제  |  A 비밀번호 변경  |  C 취소',10)
text(295,785,'문이 열려 있을 때는 D(즉시 닫기) 외의 입력을 무시합니다.',9)
c.showPage()

header(2,'비밀번호 변경 플로우차트','02  기존 비밀번호 확인 · 새 비밀번호 재확인')
X=300
box(X,122,'A 버튼 누르기',w=180,h=34,pill=True)
box(X,186,'기존 비밀번호 입력 후 #')
diamond(X,267,'기존 비밀번호 일치?')
box(106,349,'오류 안내\n변경 취소',w=150,fill=RED)
box(X,349,'새 비밀번호\n4자리 입력 후 #')
box(X,426,'새 비밀번호 재입력 후 #')
diamond(X,510,'두 입력이 일치?')
box(106,595,'기존 비밀번호 유지',w=150,fill=RED)
box(370,595,'새 비밀번호로 변경',w=190,fill=GREEN)
box(X,682,'일반 입력 모드로 복귀',w=220,h=36,pill=True)
arrow([(X,139),(X,164)])
arrow([(X,208),(X,235)])
arrow([(X,299),(X,327)],'예',(315,314))
arrow([(210,267),(106,267),(106,327)],'아니요',(154,254))
arrow([(106,371),(28,371),(28,682),(190,682)])
arrow([(X,371),(X,404)])
arrow([(X,448),(X,478)])
arrow([(210,510),(106,510),(106,573)],'아니요',(151,497))
arrow([(X,542),(X,558),(370,558),(370,573)],'예',(385,558))
arrow([(106,617),(106,650),(300,650),(300,664)])
arrow([(370,617),(370,650),(300,650)])
text(W/2,739,'변경 중: * 한 글자 삭제 / B 전체 삭제 / C 변경 취소',10)
text(W/2,761,'모든 비밀번호는 숫자 4자리입니다. D를 누르면 잠금 상태로 돌아갑니다.',9)
text(W/2,783,'변경값은 실행 중에만 유지되며, 재시작 시 3152로 초기화됩니다.',9)
c.save()

doc = pdfium.PdfDocument(str(OUT / 'doorlock-flowchart.pdf'))
for i, page in enumerate(doc):
    page.render(scale=1.5).to_pil().save(str(OUT / f'preview-{i+1}.png'))
print(f'Created {len(doc)} pages: {OUT / "doorlock-flowchart.pdf"}')
