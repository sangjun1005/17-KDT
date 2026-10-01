import { useEffect, useRef, useState } from 'react'
import { api } from './api.js'

export function Modal({ title, onClose, children }) {
  const ref = useRef(null)
  useEffect(() => {
    const dialog = ref.current
    dialog.showModal()
    return () => dialog.close()
  }, [])
  return <dialog ref={ref} onCancel={onClose} aria-labelledby="dialog-title" className="modal">
    <div className="modal-heading"><h2 id="dialog-title">{title}</h2>
      <button className="icon-button" onClick={onClose} type="button" aria-label="닫기">×</button></div>
    {children}
  </dialog>
}

export function AuthDialog({ mode, user, onClose, onSuccess, onMode }) {
  const profile = mode === 'profile'
  const signup = mode === 'signup'
  const title = profile ? '회원정보 수정' : signup ? '회원가입' : '로그인'
  const [username, setUsername] = useState(user?.username || '')
  const [nickname, setNickname] = useState(user?.nickname || '')
  const [password, setPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  async function submit(event) {
    event.preventDefault()
    if (busy) return
    setBusy(true)
    setError('')
    try {
      const body = profile ? { nickname, ...(newPassword ? { current_password: password, new_password: newPassword } : {}) }
        : { username, password, ...(signup ? { nickname } : {}) }
      const updated = await api(profile ? '/api/auth/me' : `/api/auth/${mode}`, {
        method: profile ? 'PATCH' : 'POST', body: JSON.stringify(body),
      })
      onSuccess(updated)
      onClose()
    } catch (failure) { setError(failure.message || '요청을 처리할 수 없습니다.') }
    finally { setBusy(false) }
  }
  return <Modal title={title} onClose={onClose}>
    <form onSubmit={submit} className="form-stack">
      <label>아이디<input value={username} onChange={e => setUsername(e.target.value)} required pattern="[a-zA-Z0-9_]{3,30}" maxLength={30} disabled={profile || busy} autoComplete="username" placeholder="영문·숫자·밑줄 3~30자" /></label>
      {(signup || profile) && <label>닉네임<input value={nickname} onChange={e => setNickname(e.target.value)} required maxLength={30} disabled={busy} autoComplete="nickname" placeholder="다른 회원에게 보이는 이름" /></label>}
      <label>{profile ? '현재 비밀번호' : '비밀번호'}<input type="password" value={password} onChange={e => setPassword(e.target.value)} required={!profile || Boolean(newPassword)} minLength={8} maxLength={128} disabled={busy} autoComplete={signup ? 'new-password' : 'current-password'} placeholder="8자 이상" /></label>
      {profile && <label>새 비밀번호<input type="password" value={newPassword} onChange={e => setNewPassword(e.target.value)} minLength={8} maxLength={128} disabled={busy} autoComplete="new-password" placeholder="변경할 때만 입력해 주세요" /></label>}
      {error && <p role="alert" className="error">{error}</p>}
      <button type="submit" className="primary-button" disabled={busy}>{busy ? '처리 중…' : title}</button>
      {!profile && <><p className="form-help">로그인은 최대 7일 유지됩니다.</p><button type="button" className="text-button" onClick={() => onMode(signup ? 'login' : 'signup')}>{signup ? '이미 회원이신가요? 로그인' : '처음이신가요? 회원가입'}</button></>}
      {profile && <p className="form-help">비밀번호를 변경하면 다른 기기의 로그인은 종료됩니다.</p>}
    </form>
  </Modal>
}

export function EditDialog({ story, onClose, onSave }) {
  const [title, setTitle] = useState(story.title)
  const [content, setContent] = useState(story.content)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  async function submit(event) {
    event.preventDefault()
    setBusy(true)
    try { await onSave(story.id, { title, content }); onClose() }
    catch (failure) { setError(failure.message) }
    finally { setBusy(false) }
  }
  return <Modal title="내 스토리 수정" onClose={onClose}>
    <form onSubmit={submit} className="form-stack">
      <label>제목<input required maxLength={100} value={title} onChange={e => setTitle(e.target.value)} disabled={busy} /></label>
      <label>스토리 내용<textarea required maxLength={50000} rows={10} value={content} onChange={e => setContent(e.target.value)} disabled={busy} /></label>
      {error && <p role="alert" className="error">{error}</p>}
      <button className="primary-button" disabled={busy}>{busy ? '저장 중…' : '수정 저장'}</button>
    </form>
  </Modal>
}
