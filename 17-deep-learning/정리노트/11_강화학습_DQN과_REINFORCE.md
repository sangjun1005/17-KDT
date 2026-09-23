# 11. 강화학습 — DQN과 REINFORCE

**실습 파일**: `11.ipynb`

1~10장은 전부 **정답이 주어진** 문제였다. 라벨, 마스크, 박스.
강화학습은 정답이 없다. **행동하고, 보상을 받고, 그걸로 배운다.**

| 절 | 내용 |
|---|---|
| 11-1 | 강화학습의 구성 요소 |
| 11-2 | DQN — 가치를 학습한다 |
| 11-3 | DQN이 학습에 실패한 이유 |
| 11-4 | REINFORCE — 정책을 직접 학습한다 |
| 11-5 | 두 방법의 비교 |

---

## 11-1. 강화학습의 구성 요소

### 지도학습과 무엇이 다른가

<table fit-page-width="true" header-row="true">
<tr><td></td><td>지도학습 (1~10장)</td><td>강화학습 (11장)</td></tr>
<tr><td>데이터</td><td>고정된 (입력, 정답) 쌍</td><td>행동해서 <strong>직접 만든다</strong></td></tr>
<tr><td>신호</td><td>정답과의 차이</td><td>보상 (지연될 수 있다)</td></tr>
<tr><td>목표</td><td>손실 최소화</td><td>누적 보상 최대화</td></tr>
<tr><td>i.i.d. 가정</td><td>성립</td><td><strong>깨진다</strong> (연속된 상태는 비슷하다)</td></tr>
</table>

**"정답"이 없고 "얼마나 좋았나"만 있다.** 그리고 그 평가가 한참 뒤에 온다.
바둑에서 50수째의 좋고 나쁨은 게임이 끝나야 안다. 이걸 **신용 할당 문제**라 한다.

### CartPole

```python
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]   # 4
action_size = env.action_space.n              # 2
```

```
상태 (4개) : 카트 위치, 카트 속도, 막대 각도, 막대 각속도
행동 (2개) : 왼쪽으로 밀기 / 오른쪽으로 밀기
보상       : 살아남은 스텝마다 +1
종료       : 막대가 너무 기울거나 카트가 화면을 벗어나면
최대       : 500스텝 (v1 기준)
```

**강화학습의 "Hello World"다.** 상태가 4차원이라 신경망이 아주 작아도 된다.

### 참고 — 첫 셀은 전부 주석이다

```python
# class ReinforcementLearningDemo:
#     def cartpole_example(self):
#         ...
```

`while not self.game_over` 루프로 개념을 설명하려던 의사코드다.
`update_policy`가 출력만 하고 실제로 아무것도 갱신하지 않는다.
**실행되지 않았다** (`execution_count`가 `None`).

---

## 11-2. DQN — 가치를 학습한다

### 아이디어

**"이 상태에서 이 행동을 하면 앞으로 얼마나 벌 수 있나"를 신경망으로 예측한다.**
그 값이 `Q(s, a)`다. 그다음 `Q`가 가장 큰 행동을 고른다.

```python
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

**출력이 행동 개수만큼(2개)이다.** 상태 하나를 넣으면 각 행동의 가치가 나온다.
회귀 문제이므로 **마지막에 활성화가 없다.**

### 벨만 방정식이 곧 손실이다

```python
def learn(experiences, gamma):
    states, actions, rewards, next_states, dones = experiences

    Q_targets_next = qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    Q_expected = qnetwork_local(states).gather(1, actions)

    loss = nn.MSELoss()(Q_expected, Q_targets)
    ...
```

```
Q(s, a)  =  r  +  γ · max_a' Q(s', a')
 지금 값     즉시 보상      다음 상태에서 최선의 가치
```

**목표값을 자기 자신으로 만든다.** 지도학습과 결정적으로 다른 점이다.
라벨이 밖에서 오는 게 아니라 **모델이 만든다.**

**`(1 - dones)`가 중요하다.** 에피소드가 끝난 상태에는 "다음"이 없다.
그때는 `Q_targets = rewards`가 되어야 한다.

**`.detach()`도 필수다.** 목표값 쪽으로는 기울기를 흘리지 않는다.
안 하면 "목표를 예측에 맞추는" 방향으로도 학습돼서 발산한다.

**`gather(1, actions)`가 요령이다.**
신경망은 두 행동의 Q를 다 주는데, **실제로 한 행동의 Q만** 필요하다.

```
qnetwork_local(states)          (64, 2)     두 행동의 Q
  ↓ .gather(1, actions)         (64, 1)     실제 한 행동의 Q만
```

`max(1)[0]`은 "1번 축에서 최댓값" — `[0]`은 값, `[1]`은 인덱스다.

### 두 개의 신경망

```python
qnetwork_local = QNetwork(state_size, action_size, seed)
qnetwork_target = QNetwork(state_size, action_size, seed)
```

<table fit-page-width="true" header-row="true">
<tr><td>이름</td><td>역할</td><td>갱신</td></tr>
<tr><td>local</td><td>행동 선택, 학습 대상</td><td>매 스텝</td></tr>
<tr><td>target</td><td>목표값 계산</td><td>천천히 (소프트 업데이트)</td></tr>
</table>

```python
for target_param, local_param in zip(qnetwork_target.parameters(),
                                     qnetwork_local.parameters()):
    target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
```

**`tau = 1e-3`이니 목표망은 0.1%씩만 따라간다.**

목표망이 없으면 "움직이는 과녁을 쏘는" 꼴이 된다.
예측을 바꾸면 목표도 같이 바뀌어 학습이 진동한다. 목표를 **거의 고정**시키는 장치다.

### 리플레이 버퍼

```python
memory = deque(maxlen=int(1e5))

def step(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def sample():
    experiences = random.sample(memory, k=batch_size)
    ...
```

**연속된 경험은 서로 너무 비슷하다.** 그대로 학습하면 i.i.d. 가정이 심하게 깨진다.
버퍼에 쌓아 두고 **무작위로 꺼내** 상관관계를 끊는다.

`deque(maxlen=...)`이라 가득 차면 오래된 것부터 밀려난다.

**`np.vstack`으로 배치를 만드는 게 관용구다.**

```python
states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()
```

`dones`는 불리언이라 `uint8`을 거쳐 `float`으로 만든다. 곱셈에 쓰려면 실수여야 한다.

---

## 11-3. DQN이 학습에 실패한 이유

### 결과

```
에피소드   보상
1         12.0
2          9.0
3         10.0
...
198       10.0
199       10.0
200        9.0
```

**200 에피소드 내내 8~12에 머문다.** 전혀 학습되지 않았다.

**이건 무작위보다도 나쁘다.** CartPole에서 동전 던지듯 행동하면 평균 20스텝쯤 버틴다.
9~10은 **한쪽으로만 계속 미는** 정책의 성적이다.

### 원인 — 탐험이 없다

```python
with torch.no_grad():
    action_values = qnetwork_local(state_tensor)
action = np.argmax(action_values.cpu().data.numpy())     # 항상 최댓값
```

**ε-greedy가 없다.** 처음부터 끝까지 **순수 탐욕(greedy)**으로 행동한다.

```
초기 Q 신경망은 난수다
  → 거의 항상 같은 행동을 고른다 (예: 늘 오른쪽)
  → 막대가 10스텝 만에 쓰러진다
  → 다양한 상태를 경험하지 못한다
  → 버퍼에 비슷한 경험만 쌓인다
  → Q가 개선되지 않는다
  → 여전히 같은 행동을 고른다        ← 악순환
```

**탐험(exploration)과 활용(exploitation)의 균형이 강화학습의 근본 문제다.**
지도학습에는 없는 개념이다. 데이터가 고정돼 있지 않고 **내가 만들기** 때문이다.

고치려면 ε-greedy를 넣는다. (원본은 그대로 두었다.)

```python
eps = max(eps_end, eps_decay * eps)          # 1.0에서 시작해 0.01까지 감소

if random.random() > eps:
    action = np.argmax(action_values.cpu().data.numpy())   # 활용
else:
    action = random.choice(np.arange(action_size))         # 탐험
```

```
학습 초반 : ε = 1.0   →  거의 무작위로 돌아다니며 경험을 모은다
학습 후반 : ε = 0.01  →  배운 대로 행동한다
```

### 다른 아쉬운 점들

<table fit-page-width="true" header-row="true">
<tr><td>항목</td><td>이 코드</td><td>보통</td></tr>
<tr><td>ε-greedy</td><td><strong>없음</strong></td><td>1.0 → 0.01 감쇠</td></tr>
<tr><td>목표망 갱신</td><td>에피소드마다 1회</td><td>학습 스텝마다</td></tr>
<tr><td>학습 빈도</td><td>매 스텝</td><td>보통 4스텝마다</td></tr>
<tr><td>점수 추적</td><td>매 에피소드 출력</td><td>최근 100개 평균</td></tr>
</table>

**목표망 갱신이 에피소드마다 1회인 게 특히 느리다.**
`tau=1e-3`은 "매 학습 스텝" 기준의 값이다.
에피소드가 10스텝이면 목표망은 100배 느리게 따라간다. 사실상 초기 난수에 고정된다.

### 참고 — gym API 경고

```
Gym has been unmaintained since 2022 and does not support NumPy 2.0
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym
DeprecationWarning: WARN: Initializing wrapper in old step API which returns
one bool instead of two.
```

**gym은 Gymnasium으로 넘어갔다.** API도 바뀌었다.

```python
# 옛 gym
state = env.reset()
next_state, reward, done, info = env.step(action)

# Gymnasium
state, info = env.reset()
next_state, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

**`terminated`와 `truncated`를 나눈 게 중요한 변화다.**

```
terminated : 진짜로 실패했다 (막대가 쓰러졌다)      → Q_target에 다음 상태를 안 쓴다
truncated  : 시간 제한에 걸렸다 (500스텝 도달)      → 다음 상태를 써야 한다
```

옛 API는 둘을 `done` 하나로 뭉뚱그렸다. 시간 제한으로 끝난 걸 "실패"로 처리하면
**끝까지 버틴 좋은 정책을 벌주게 된다.**

---

## 11-4. REINFORCE — 정책을 직접 학습한다

### 가치가 아니라 확률을 낸다

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return torch.softmax(self.fc2(x), dim=-1)      # ← 확률
```

**DQN의 Q 신경망과 출력이 다르다.**

```
DQN       : 출력 = 각 행동의 가치 (실수)   → argmax로 행동 결정
REINFORCE : 출력 = 각 행동의 확률         → 샘플링으로 행동 결정
```

### 샘플링이 곧 탐험이다

```python
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy_net(state)
    action = torch.multinomial(probs, num_samples=1)
    return action.item(), torch.log(probs[0, action.item()])
```

**`torch.multinomial`이 확률대로 뽑는다.** `argmax`가 아니다.

```
probs = [0.7, 0.3]  →  70% 확률로 0번, 30% 확률로 1번
```

**탐험 장치를 따로 안 붙여도 된다.** 확률적 정책 자체가 탐험이다.
11-3에서 DQN이 막힌 지점을 구조적으로 피한다.

**`log_prob`를 같이 돌려주는 게 핵심이다.** 손실 계산에 그게 필요하다.

### 할인 누적 보상

```python
def reinforce_update(episode_rewards, episode_log_probs, gamma=0.99):
    R = 0
    returns = []
    for r in episode_rewards[::-1]:          # 뒤에서부터
        R = r + gamma * R
        returns.insert(0, R)
```

**뒤에서부터 거슬러 올라가며 누적한다.**

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

한 번의 역방향 순회로 모든 시점의 `G_t`가 나온다.
앞에서부터 하면 시점마다 뒤쪽을 다시 다 더해야 해서 제곱 시간이 걸린다.

**`gamma = 0.99`가 "먼 미래를 얼마나 중요하게 볼 것인가"다.**

```
γ = 0    →  당장의 보상만 본다 (근시안)
γ = 0.99 →  100스텝쯤 뒤까지 본다
γ = 1    →  무한히 먼 미래도 똑같이 (수렴이 안 될 수 있다)
```

### 기준선 — 표준화

```python
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

**이 한 줄이 REINFORCE를 돌아가게 만든다.**

표준화 전에는 모든 `G_t`가 양수다(보상이 +1뿐이니까).
그러면 **모든 행동의 확률을 다 올리려** 해서 방향이 흐려진다.

표준화하면 평균보다 좋았던 시점은 양수, 나빴던 시점은 음수가 된다.

```
G_t > 평균  →  그 행동의 확률을 올린다
G_t < 평균  →  그 행동의 확률을 내린다
```

**이게 기준선(baseline)의 역할이다.** 분산을 크게 줄인다.
`1e-8`은 표준편차가 0일 때(에피소드가 1스텝)의 0 나눗셈 방지다.

### 정책 경사 손실

```python
loss = 0
for log_prob, R in zip(episode_log_probs, returns):
    loss -= log_prob * R

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

```
loss = - Σ log π(a_t | s_t) · G_t
```

**부호가 음수인 이유** — 옵티마이저는 최소화만 한다.
우리가 원하는 건 기대 보상의 **최대화**라서 부호를 뒤집는다.

**`log_prob`를 쓰는 이유** — `log`의 미분이 `1/π`라
확률이 작은(드물게 고른) 행동에 더 큰 기울기가 간다.
수학적으로는 정책 경사 정리에서 그대로 나온다.

### 결과 — 학습된다

```
에피소드   보상    손실
0         37.0   -0.065
50        23.0    0.212
100      109.0   -0.010
150      500.0   -8.609      ← 최대치 도달
200      500.0   -9.480
250      254.0   -6.648
300      500.0   15.367
...
550      500.0   -7.266
600      500.0   -0.577
650      186.0    1.158
700      500.0    6.218
750       30.0   -4.351      ← 무너짐
800      158.0   -5.317
850       21.0    3.588
900       43.0   -0.133
950       48.0   -1.322
```

**150 에피소드 만에 500(최대치)에 도달한다.** DQN이 200 에피소드 내내 10이었던 것과 대조적이다.

**그런데 안정적이지 않다.** 500까지 갔다가 20으로 떨어지길 반복한다.

REINFORCE의 고질적인 문제다.

<table fit-page-width="true" header-row="true">
<tr><td>문제</td><td>이유</td></tr>
<tr><td><strong>높은 분산</strong></td><td>에피소드 하나로 기울기를 만든다. 운에 크게 좌우된다</td></tr>
<tr><td>표본 비효율</td><td>쓴 에피소드를 버린다 (DQN은 버퍼에 재사용)</td></tr>
<tr><td>파국적 갱신</td><td>운 나쁜 에피소드 하나가 정책을 크게 망가뜨린다</td></tr>
</table>

**`lr = 0.01`이 크다.** 700에서 500이었다가 750에서 30으로 떨어진 건
한 번의 큰 갱신이 정책을 무너뜨린 것으로 보인다.

**손실값의 부호나 크기는 성능과 무관하다.**
`-8.6`도 `+15.4`도 둘 다 보상 500일 때 나왔다.
정책 경사의 손실은 "얼마나 틀렸나"가 아니라 **"이번에 얼마나 크게 밀었나"**다.
**보상 곡선을 봐야 한다.**

### 개선 방향

```
REINFORCE
   ↓  가치 함수를 기준선으로 (분산 감소)
Actor-Critic
   ↓  갱신 크기를 제한 (파국적 갱신 방지)
PPO  ← 요즘 표준
```

**PPO가 지금 가장 널리 쓰인다.** 로봇 제어부터 LLM의 RLHF까지.

---

## 11-5. 두 방법의 비교

<table fit-page-width="true" header-row="true">
<tr><td></td><td>DQN (가치 기반)</td><td>REINFORCE (정책 기반)</td></tr>
<tr><td>학습 대상</td><td>Q(s, a)</td><td>π(a|s)</td></tr>
<tr><td>출력</td><td>행동별 가치</td><td>행동별 확률</td></tr>
<tr><td>행동 선택</td><td>argmax + ε-greedy</td><td>확률 샘플링</td></tr>
<tr><td>탐험</td><td><strong>따로 넣어야 한다</strong></td><td>내장</td></tr>
<tr><td>갱신 시점</td><td>매 스텝</td><td>에피소드 끝</td></tr>
<tr><td>경험 재사용</td><td>O (리플레이 버퍼)</td><td>X</td></tr>
<tr><td>연속 행동</td><td>어렵다</td><td><strong>가능</strong></td></tr>
<tr><td>분산</td><td>낮음</td><td><strong>높음</strong></td></tr>
<tr><td>이 노트북 결과</td><td>10 (실패)</td><td>최대 500 (불안정)</td></tr>
</table>

**이 노트북의 결과가 "탐험이 전부"라는 걸 보여 준다.**
DQN이 나쁜 알고리즘이라서 실패한 게 아니다. ε-greedy 한 줄이 빠져서다.
REINFORCE는 샘플링이 그 역할을 자동으로 해 줬다.

**연속 행동에서는 정책 기반이 유리하다.**
로봇 관절 각도처럼 행동이 실수면 `argmax`를 쓸 수 없다.
정책망은 가우시안의 평균·분산을 내놓으면 된다.

---

## 이 장 정리

### 흐름

```
환경             11-1   상태 4개, 행동 2개, 보상 +1/스텝
   ↓
DQN             11-2   Q(s,a) 회귀. 벨만 방정식이 목표값
   ↓
실패 원인         11-3   ε-greedy 없음 → 탐험이 0
   ↓
REINFORCE       11-4   π(a|s) 샘플링. 표준화한 G_t로 가중
   ↓
비교             11-5   가치 기반 vs 정책 기반
```

### DQN 필수 요소 다섯

```
1. 리플레이 버퍼      경험의 상관관계를 끊는다
2. 목표망            움직이는 과녁 문제를 막는다
3. ε-greedy         탐험 ← 이 노트북에 빠진 것
4. (1 - done)       종료 상태 처리
5. .detach()        목표값으로 기울기를 안 흘린다
```

### REINFORCE 4줄

```python
# returns = 뒤에서부터 누적한 G_t
returns = torch.tensor(returns)
returns = (returns - returns.mean()) / (returns.std() + 1e-8)   # 기준선
loss = -sum(log_prob * R for log_prob, R in zip(log_probs, returns))
loss.backward()
```

### 학습이 되는지 확인하는 법

<table fit-page-width="true" header-row="true">
<tr><td>영역</td><td>무엇을 보나</td><td>무작위 수준</td></tr>
<tr><td>분류</td><td>손실</td><td>ln(클래스 수)</td></tr>
<tr><td>분할</td><td>Dice 손실</td><td>0.5</td></tr>
<tr><td>언어 생성</td><td>손실</td><td>ln(어휘 크기)</td></tr>
<tr><td><strong>강화학습</strong></td><td><strong>에피소드 보상</strong></td><td>CartPole은 약 20</td></tr>
</table>

**강화학습에서 손실은 학습 진척도를 나타내지 않는다.**
DQN의 MSE는 목표가 계속 바뀌어서, REINFORCE의 손실은 부호가 갱신 방향이라서 그렇다.
**보상 곡선만 본다.** 그것도 최근 100 에피소드 평균으로.

### 자주 틀리는 것

- **ε-greedy를 빼먹는다** (이 노트북의 DQN이 그래서 실패했다)
- 목표망 없이 같은 망으로 목표를 만든다 (발산한다)
- 목표값에 `.detach()`를 안 한다
- `(1 - done)`을 빼먹어 종료 상태에서도 미래 가치를 더한다
- `gather`로 실제 행동의 Q를 안 고르고 전체를 쓴다
- REINFORCE에서 `returns`를 표준화하지 않는다 (분산이 폭발한다)
- 손실 부호를 안 뒤집는다 (보상을 최소화하게 된다)
- **손실 곡선으로 강화학습 성능을 판단한다**
- 옛 gym API의 `done`을 그대로 쓴다 (`terminated` / `truncated` 구분 필요)
- 에피소드 하나의 보상으로 성능을 판단한다 (100개 평균을 본다)
