import { useEffect, useState } from 'react'
import { api } from './api.js'
import { AuthDialog, EditDialog, Modal } from './Dialogs.jsx'
import StoryCard from './StoryCard.jsx'

const EMPTY_LIST = { items: [], total: 0, total_pages: 0 }
const MAX_INPUT_TOKENS = 56
// 브라우저에서는 토큰을 셀 수 없어 글자 수로 대략 막고, 최종 판정은 서버가 토큰 수로 한다.
const MAX_INPUT_CHARS = 300

export default function App() {
  const [user, setUser] = useState(null)
  const [authReady, setAuthReady] = useState(false)
  const [authMode, setAuthMode] = useState(null)
  const [prompt, setPrompt] = useState('')
  const [story, setStory] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [listing, setListing] = useState(EMPTY_LIST)
  const [page, setPage] = useState(1)
  const [revision, setRevision] = useState(0)
  const [listBusy, setListBusy] = useState(false)
  const [pendingId, setPendingId] = useState(null)
  const [editing, setEditing] = useState(null)
  const [deleting, setDeleting] = useState(null)

  useEffect(() => {
    api('/api/auth/me').then(setUser).catch(failure => {
      if (failure.status !== 401) setError('서버에 연결할 수 없습니다.')
    }).finally(() => setAuthReady(true))
  }, [])

  useEffect(() => {
    if (!user) return
    let active = true
    setListBusy(true)
    api(`/api/stories?page=${page}`).then(data => {
      if (!active) return
      if (page > Math.max(1, data.total_pages)) setPage(Math.max(1, data.total_pages))
      else setListing(data)
    }).catch(failure => { if (active) handleFailure(failure) })
      .finally(() => { if (active) setListBusy(false) })
    return () => { active = false }
  }, [user?.id, page, revision])

  function resetSignedOut() {
    setUser(null); setStory(null); setListing(EMPTY_LIST); setPage(1)
    setEditing(null); setDeleting(null)
  }
  function handleFailure(failure) {
    setError(failure.message || '서버에 연결할 수 없습니다.')
    if (failure.status === 401) { resetSignedOut(); setAuthMode('login') }
  }
  function signedIn(updated) {
    setUser(updated); setError(''); setNotice(''); setRevision(value => value + 1)
  }
  async function logout() {
    try { await api('/api/auth/logout', { method: 'POST' }) }
    catch (failure) { if (failure.status !== 401) return handleFailure(failure) }
    resetSignedOut(); setError(''); setNotice('')
  }
  async function generate(event) {
    event.preventDefault()
    if (!user) return setAuthMode('login')
    if (busy || !prompt.trim()) return
    setBusy(true); setError(''); setNotice(''); setStory(null)
    try {
      const result = await api('/api/stories/generate', { method: 'POST', body: JSON.stringify({ prompt }) })
      setStory(result); setPage(1); setRevision(value => value + 1)
      setNotice('스토리가 기록에 저장되었습니다.')
    } catch (failure) { handleFailure(failure) }
    finally { setBusy(false) }
  }
  function updateVisibleStory(updated) {
    setListing(previous => ({ ...previous, items: previous.items.map(item => item.id === updated.id ? updated : item) }))
    setStory(previous => previous?.id === updated.id ? { ...previous, ...updated } : previous)
  }
  async function toggleLike(item) {
    if (pendingId !== null) return
    setPendingId(item.id); setError('')
    try { updateVisibleStory(await api(`/api/stories/${item.id}/like`, { method: item.liked ? 'DELETE' : 'PUT' })) }
    catch (failure) { handleFailure(failure) }
    finally { setPendingId(null) }
  }
  async function saveEdit(id, data) {
    try {
      updateVisibleStory(await api(`/api/stories/${id}`, { method: 'PATCH', body: JSON.stringify(data) }))
      setNotice('스토리를 수정했습니다.')
    } catch (failure) { handleFailure(failure); throw failure }
  }
  async function confirmDelete() {
    if (pendingId !== null) return
    setPendingId(deleting.id)
    try {
      await api(`/api/stories/${deleting.id}`, { method: 'DELETE' })
      if (story?.id === deleting.id) setStory(null)
      setDeleting(null); setRevision(value => value + 1); setNotice('스토리를 삭제했습니다.')
    } catch (failure) { handleFailure(failure); setDeleting(null) }
    finally { setPendingId(null) }
  }
  const cardActions = { onLike: toggleLike, onEdit: setEditing, onDelete: setDeleting, busy: pendingId !== null }

  return <div className="site-shell">
    <div className="top-bar"><div className="top-inner">
      {user ? <><span><b>{user.nickname}</b>님</span><button className="top-link" onClick={() => setAuthMode('profile')}>회원정보 수정</button><button className="top-link" onClick={logout} disabled={busy}>로그아웃</button></>
        : <><button className="top-link" onClick={() => setAuthMode('signup')}>회원가입</button><button className="top-link" onClick={() => setAuthMode('login')}>로그인</button></>}
    </div></div>

    <header className="hero">
      <h1 className="brand"><span className="brand-mark" aria-hidden="true">S</span><span>STORYBOT</span></h1>
      <form className="search-bar" onSubmit={generate}>
        <label className="sr-only" htmlFor="prompt">이야기의 시작 문장</label>
        <input id="prompt" value={prompt} onChange={e => setPrompt(e.target.value)} maxLength={MAX_INPUT_CHARS}
          placeholder={user ? 'Once upon a time, there was a little girl named Lily.' : '로그인 후 이야기를 시작할 수 있습니다'}
          disabled={!user || busy} autoComplete="off" spellCheck={false} />
        <button className="search-button" disabled={!authReady || busy || (user && !prompt.trim())}>
          <svg aria-hidden="true" viewBox="0 0 24 24" width="22" height="22"><circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" strokeWidth="2.6" /><path d="M15.5 15.5 21 21" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" /></svg>
          <span>{busy ? '생성 중...' : user ? '생성' : '로그인'}</span></button>
      </form>
      <p className="hero-help">시작 문장은 최대 {MAX_INPUT_TOKENS}토큰(영어 약 230자)까지 입력할 수 있어요. 영어로 쓰면 더 자연스러운 이야기가 나옵니다. <span className="char-count">{prompt.length} / {MAX_INPUT_CHARS}자</span></p>
    </header>

    <main className="portal">
      <div className="main-column">
        {error && <p className="error" role="alert">{error}</p>}
        {notice && <p className="notice" role="status">{notice}</p>}
        {!authReady ? <p className="loading">로그인 상태를 확인하고 있습니다...</p>
          : !user ? <section className="card locked">
            <span className="lock-icon" aria-hidden="true">🔒</span>
            <h2>로그인 후 이용할 수 있습니다</h2>
            <p>스토리 만들기, 스토리 기록 보기, 좋아요, 수정·삭제는 회원만 이용할 수 있어요.</p>
            <button className="primary-button" onClick={() => setAuthMode('login')}>로그인</button>
          </section>
          : <>
            <section className="card result-card" aria-busy={busy}>
              <div className="card-heading"><h2>새로 만든 이야기</h2><span>최대 200토큰 생성</span></div>
              <div aria-live="polite">{busy ? <p className="empty">스토리봇이 이야기를 이어 쓰고 있습니다...</p>
                : story ? <StoryCard story={story} {...cardActions} />
                  : <p className="empty">위 입력창에 시작 문장을 넣고 <b>생성</b>을 눌러 보세요.</p>}</div>
            </section>

            <section className="card records-card">
              <div className="card-heading"><h2>스토리 기록</h2><span>총 {listing.total}개</span></div>
              {listBusy ? <p className="empty">기록을 불러오고 있습니다...</p>
                : listing.items.length === 0 ? <p className="empty">아직 저장된 이야기가 없습니다.</p>
                  : listing.items.map(item => <StoryCard key={item.id} story={item} {...cardActions} />)}
              {listing.total_pages > 0 && <nav className="pagination" aria-label="기록 페이지">
                <button disabled={page <= 1 || listBusy} onClick={() => setPage(value => value - 1)}>‹ 이전</button>
                {Array.from({ length: listing.total_pages }, (_, i) => i + 1)
                  .filter(n => Math.abs(n - page) <= 2 || n === 1 || n === listing.total_pages)
                  .map((n, i, list) => <span key={n} className="page-group">
                    {i > 0 && n - list[i - 1] > 1 && <span className="ellipsis">…</span>}
                    <button className={n === page ? 'current' : ''} aria-current={n === page ? 'page' : undefined} disabled={listBusy} onClick={() => setPage(n)}>{n}</button>
                  </span>)}
                <button disabled={page >= listing.total_pages || listBusy} onClick={() => setPage(value => value + 1)}>다음 ›</button>
              </nav>}
            </section>
          </>}
      </div>

      <aside className="sidebar">
        <section className="card account-card">{!authReady ? <p>확인 중...</p> : user ? <>
          <div className="account-intro"><span className="large-avatar" aria-hidden="true">{user.nickname.slice(0, 1)}</span>
            <div><strong>{user.nickname}님</strong><p>@{user.username}</p></div></div>
          <div className="account-links">
            <button className="text-button" onClick={() => setAuthMode('profile')}>회원정보 수정</button>
            <span aria-hidden="true">|</span>
            <button className="text-button" onClick={logout} disabled={busy}>로그아웃</button>
          </div>
        </> : <>
          <p>스토리봇을 더 안전하고 편리하게 이용하세요.</p>
          <button className="login-button" onClick={() => setAuthMode('login')}><b>STORYBOT</b> 로그인</button>
          <div className="account-links"><button className="text-button" onClick={() => setAuthMode('signup')}>회원가입</button></div>
        </>}</section>
        <section className="card guide-card">
          <h2>이용 안내</h2>
          <ol><li>시작 문장을 입력하고 생성을 눌러요.</li><li>스토리봇이 이야기를 이어 씁니다.</li><li>기록에서 <b>좋아요</b>를 눌러 보세요.</li><li>내 스토리는 <b>수정</b>·<b>삭제</b>할 수 있어요.</li></ol>
        </section>
      </aside>
    </main>
    <footer className="site-footer"><strong>STORYBOT</strong> 한 줄의 시작, 나만의 이야기</footer>

    {authMode && <AuthDialog key={authMode} mode={authMode} user={user} onClose={() => setAuthMode(null)} onMode={setAuthMode} onSuccess={signedIn} />}
    {editing && <EditDialog story={editing} onClose={() => setEditing(null)} onSave={saveEdit} />}
    {deleting && <Modal title="스토리를 삭제할까요?" onClose={() => setDeleting(null)}>
      <p className="delete-description">「{deleting.title}」와 연결된 좋아요도 함께 삭제됩니다. 삭제한 스토리는 되돌릴 수 없습니다.</p>
      <div className="confirm-actions"><button className="secondary-button" disabled={pendingId !== null} onClick={() => setDeleting(null)}>취소</button>
        <button className="danger-button" disabled={pendingId !== null} onClick={confirmDelete}>{pendingId !== null ? '삭제 중...' : '삭제'}</button></div>
    </Modal>}
  </div>
}
