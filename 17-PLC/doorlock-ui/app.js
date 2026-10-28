import { DoorLock } from './logic.mjs';

const $ = id => document.getElementById(id);
const scene = $('scene');
let cameraStatus = 'off';
let seenSequence = 0;
let polling = false;
let frameURL = null;
let demo = false;
let demoPending = false;
let cameraRequestPending = false;
const slots = [...$('pinSlots').children];
const passwordDialog = $('passwordDialog');
const passwordStorageKey = 'onjeon-demo-pin';
let savedPassword = '3152';
try { savedPassword = localStorage.getItem(passwordStorageKey) || '3152'; } catch {}

function addLog(message, isDemo = false) {
  const li = document.createElement('li');
  const dot = document.createElement('span');
  dot.className = 'event-dot' + (isDemo ? ' demo' : '');
  const text = document.createElement('span');
  text.textContent = message;
  const time = document.createElement('time');
  time.textContent = new Date().toLocaleTimeString('ko-KR', {
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
  });
  li.append(dot, text, time);
  $('activityLog').prepend(li);
  while ($('activityLog').children.length > 5) $('activityLog').lastElementChild.remove();
}

const labels = {
  locked: ['안전하게 잠겨 있어요', '등록된 손동작으로\n우리 집의 문을 열어 주세요.', '현관문 잠김', 'LOCKED'],
  opening: ['반가워요, 어서 오세요', '인증을 확인했습니다.\n현관문이 열리고 있습니다.', '현관문 여는 중', 'WELCOME'],
  open: ['집에 오신 걸 환영해요', '편안한 하루의 시작.\n5초 후 문이 자동으로 닫힙니다.', '현관문 열림', 'OPEN'],
  closing: ['문을 안전하게 닫아요', '문이 닫힌 후\n자동으로 잠금 상태가 됩니다.', '현관문 닫는 중', 'CLOSING'],
  denied: ['순서를 다시 확인해 주세요', '등록된 손동작과 다릅니다.\n잠시 후 다시 입력해 주세요.', '인증 실패 · 잠금 유지', 'RETRY']
};

const lock = new DoorLock({ password: savedPassword, onChange(event) {
  render();
  const messages = {
    accepted: demo ? '시연 · 열림 모션을 시작합니다.' : '손가락 인증 성공 · 현관문을 엽니다.',
    denied: '인증 실패 · 현관문은 잠긴 상태를 유지합니다.',
    open: '현관문 열림 · 5초 후 자동으로 닫힙니다.',
    closing: '현관문을 닫고 있습니다.',
    locked: '현관문 잠금 완료'
  };
  if (messages[event]) addLog(messages[event], demo);
  if (event === 'locked') { demo = false; updateButtons(); }
} });

function updateButtons() {
  $('demoButton').disabled = passwordDialog.open || demoPending || lock.phase !== 'locked' || cameraStatus === 'on' || cameraStatus === 'starting';
  $('cameraButton').disabled = passwordDialog.open || demoPending || demo || cameraStatus === 'starting' || cameraRequestPending;
  $('closeButton').disabled = !['open', 'opening'].includes(lock.phase);
  const canEdit = !passwordDialog.open && ['locked', 'denied'].includes(lock.phase) && lock.entered.length > 0 && !demo && !demoPending;
  $('backspaceButton').disabled = !canEdit;
  $('clearButton').disabled = !canEdit;
  $('passwordButton').disabled = passwordDialog.open || lock.phase !== 'locked' || demo || demoPending;
}

function render() {
  scene.className = 'scene ' + lock.phase;
  const [title, description, caption, screen] = labels[lock.phase];
  $('statusTitle').textContent = title;
  $('statusDescription').textContent = description;
  $('statusDescription').style.whiteSpace = 'pre-line';
  $('doorCaption').textContent = caption;
  $('lockScreen').textContent = screen;
  $('statusIcon').textContent = ['opening', 'open'].includes(lock.phase) ? '⌂' : lock.phase === 'denied' ? '!' : '⌑';
  $('statusIcon').classList.toggle('denied', lock.phase === 'denied');
  $('doorStateDot').classList.toggle('live', lock.phase === 'open');
  slots.forEach((slot, i) => slot.classList.toggle('filled', i < lock.entered.length));
  $('pinSlots').setAttribute('aria-label', `입력한 비밀번호 ${lock.entered.length}자리`);
  updateButtons();
}

function refreshTimer() {
  lock.tick();
  if (lock.phase === 'open') {
    $('timerLabel').textContent = `${lock.secondsLeft}초 후 닫힘`;
    $('countdownProgress').style.width = `${Math.max(0, (lock.deadline - performance.now()) / lock.autoCloseMs * 100)}%`;
  } else {
    $('timerLabel').textContent = lock.phase === 'closing' ? '닫는 중' : '5초';
    $('countdownProgress').style.width = '0%';
  }
}

$('closeButton').addEventListener('click', () => lock.close());
$('passwordButton').addEventListener('click', () => {
  if ($('passwordButton').disabled) return;
  $('passwordForm').reset();
  $('passwordError').textContent = '';
  lock.clear();
  passwordDialog.showModal();
  updateButtons();
});
$('passwordCancel').addEventListener('click', () => passwordDialog.close());
passwordDialog.addEventListener('close', () => {
  $('passwordForm').reset();
  $('passwordError').textContent = '';
  updateButtons();
});
$('passwordForm').addEventListener('submit', event => {
  event.preventDefault();
  const error = lock.changePassword(
    $('currentPassword').value,
    $('newPassword').value,
    $('confirmPassword').value,
    next => localStorage.setItem(passwordStorageKey, next)
  );
  $('passwordError').textContent = error;
  if (error) return;
  passwordDialog.close();
  addLog('비밀번호 변경 완료 · 이 브라우저에 저장되었습니다.');
  $('statusDescription').textContent = '비밀번호가 변경되었습니다.\n새 손동작 순서로 입력해 주세요.';
});
$('backspaceButton').addEventListener('click', () => {
  if (!$('backspaceButton').disabled) lock.backspace();
});
$('clearButton').addEventListener('click', () => {
  if (!$('clearButton').disabled) lock.clear();
});
$('demoButton').addEventListener('click', async () => {
  if ($('demoButton').disabled) return;
  demo = true;
  demoPending = true;
  lock.clear();
  updateButtons();
  for (const digit of lock.password.split('').map(Number)) {
    await new Promise(resolve => setTimeout(resolve, 220));
    lock.input(digit);
  }
  demoPending = false;
  updateButtons();
});

function showCamera(state) {
  const previous = cameraStatus;
  cameraStatus = state.status;
  $('cameraTag').textContent = {on: 'LIVE · LOCAL', off: 'STANDBY', starting: 'CONNECTING', error: 'UNAVAILABLE'}[cameraStatus] || 'OFFLINE';
  $('cameraDot').classList.toggle('live', cameraStatus === 'on');
  $('cameraButton').innerHTML = cameraStatus === 'on' ? '카메라 끄기 <span>×</span>' : cameraStatus === 'starting' ? '카메라 연결 중…' : '카메라 켜기 <span>↗</span>';
  $('cameraNote').textContent = state.message || '영상은 이 PC에서만 처리됩니다.';
  $('cameraNote').classList.toggle('error', ['error', 'offline'].includes(cameraStatus));
  $('fingerCount').textContent = state.fingers ?? '—';
  $('holdProgress').style.width = `${(state.progress || 0) * 100}%`;
  $('holdLabel').textContent = state.armed === false ? '손을 빼면 다음 입력' : '1초 유지하여 입력';
  $('cameraOverlay').textContent = state.fingers === null ? '손 인식 대기' : `손가락 ${state.fingers}개`;
  if (cameraStatus !== 'on') {
    $('cameraFrame').hidden = true;
    $('cameraEmpty').hidden = false;
    $('cameraOverlay').hidden = true;
  }
  if (previous === 'on' && cameraStatus !== 'on') lock.clear();
  if (previous !== 'on' && cameraStatus === 'on') addLog('카메라 연결 완료 · 손동작 입력 대기');
  updateButtons();
}

$('cameraButton').addEventListener('click', async () => {
  if ($('cameraButton').disabled) return;
  cameraRequestPending = true;
  updateButtons();
  try {
    const snapshot = await fetch('/api/state', {cache: 'no-store'}).then(r => r.json());
    seenSequence = snapshot.sequence;
    const action = cameraStatus === 'on' ? 'stop' : 'start';
    const result = await fetch(`/api/camera/${action}`, {method: 'POST'});
    if (!result.ok) throw new Error('카메라 연결 요청을 처리하지 못했습니다.');
    lock.clear();
    showCamera(await result.json());
    if (action === 'stop') addLog('카메라를 껐습니다.');
  } catch (error) {
    showCamera({status: 'offline', fingers: null, message: '실행.cmd로 서버를 실행한 뒤 http://127.0.0.1:8765에서 열어 주세요.'});
  } finally {
    cameraRequestPending = false;
    updateButtons();
  }
});

async function pollCamera() {
  if (polling || document.hidden) return;
  polling = true;
  try {
    const response = await fetch('/api/state', {cache: 'no-store', signal: AbortSignal.timeout(2500)});
    if (!response.ok) throw new Error('Server unavailable');
    const state = await response.json();
    showCamera(state);
    for (const event of state.events || []) {
      if (event.sequence > seenSequence) {
        seenSequence = event.sequence;
        if (state.status === 'on' && !demo && !demoPending && !passwordDialog.open) lock.input(event.digit);
      }
    }
    if (state.status === 'on') {
      const frame = await fetch('/api/frame.jpg', {cache: 'no-store', signal: AbortSignal.timeout(2500)});
      if (frame.ok) {
        const nextURL = URL.createObjectURL(await frame.blob());
        $('cameraFrame').src = nextURL;
        if (frameURL) URL.revokeObjectURL(frameURL);
        frameURL = nextURL;
        $('cameraFrame').hidden = false;
        $('cameraEmpty').hidden = true;
        $('cameraOverlay').hidden = false;
      }
    }
  } catch {
    showCamera({status: 'offline', fingers: null, message: '서버 연결 대기 · 실행.cmd를 실행해 주세요.'});
  } finally { polling = false; }
}

// Ignore gestures made while this page was hidden, rather than replaying them as a PIN.
document.addEventListener('visibilitychange', async () => {
  if (!document.hidden) {
    try { seenSequence = (await fetch('/api/state').then(r => r.json())).sequence; } catch {}
    lock.clear();
  }
});
window.addEventListener('pagehide', () => {
  if (['on', 'starting'].includes(cameraStatus)) {
    fetch('/api/camera/stop', {method: 'POST', keepalive: true}).catch(() => {});
  }
  if (frameURL) URL.revokeObjectURL(frameURL);
});
render();
setInterval(refreshTimer, 50);
setInterval(pollCamera, 140);
pollCamera();
