export class DoorLock {
  constructor({ now = () => performance.now(), onChange = () => {}, password = '3152' } = {}) {
    this.now = now;
    this.onChange = onChange;
    this.phase = 'locked';
    this.entered = [];
    this.deadline = 0;
    this.password = typeof password === 'string' && /^[0-5]{4}$/.test(password) ? password : '3152';
    this.autoCloseMs = 5000;
  }
  input(digit) {
    if (this.phase !== 'locked' || !Number.isInteger(digit) || digit < 0 || digit > 5) return;
    this.entered.push(String(digit));
    if (this.entered.length < 4) { this.onChange('digit'); return; }
    if (this.entered.join('') === this.password) {
      this.phase = 'opening';
      this.deadline = this.now() + 1300;
      this.onChange('accepted');
    } else {
      this.phase = 'denied';
      this.deadline = this.now() + 1400;
      this.onChange('denied');
    }
  }
  clear() {
    if (!['locked', 'denied'].includes(this.phase)) return;
    this.phase = 'locked';
    this.deadline = 0;
    this.entered = [];
    this.onChange('clear');
  }
  backspace() {
    if (!['locked', 'denied'].includes(this.phase) || this.entered.length === 0) return;
    this.phase = 'locked';
    this.deadline = 0;
    this.entered.pop();
    this.onChange('backspace');
  }
  changePassword(current, next, confirmation, save = () => {}) {
    if (this.phase !== 'locked') return '문이 잠긴 상태에서 변경해 주세요.';
    if (current !== this.password) return '현재 비밀번호가 일치하지 않습니다.';
    if (typeof next !== 'string' || !/^[0-5]{4}$/.test(next)) return '새 비밀번호는 0~5 숫자로 된 네 자리여야 합니다.';
    if (next !== confirmation) return '새 비밀번호와 확인 입력이 다릅니다.';
    if (next === current) return '현재와 다른 비밀번호를 입력해 주세요.';
    try { save(next); } catch { return '브라우저에 저장하지 못했습니다. 저장 설정을 확인해 주세요.'; }
    this.password = next;
    this.clear();
    return '';
  }
  close() {
    if (this.phase === 'open' || this.phase === 'opening') {
      this.phase = 'closing';
      this.deadline = this.now() + 1300;
      this.onChange('closing');
    }
  }
  tick() {
    if (!this.deadline || this.now() < this.deadline) return;
    if (this.phase === 'opening') {
      this.phase = 'open';
      this.deadline = this.now() + this.autoCloseMs;
      this.onChange('open');
    } else if (this.phase === 'open') {
      this.close();
    } else if (this.phase === 'closing' || this.phase === 'denied') {
      this.phase = 'locked';
      this.entered = [];
      this.deadline = 0;
      this.onChange('locked');
    }
  }
  get secondsLeft() { return Math.max(0, Math.ceil((this.deadline - this.now()) / 1000)); }
}
