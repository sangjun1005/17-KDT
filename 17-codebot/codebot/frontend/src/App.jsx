import { useEffect, useState } from 'react'
import { api } from './api.js'
import { AuthDialog, EditDialog, Modal } from './Dialogs.jsx'
import StoryCard from './StoryCard.jsx'

const EMPTY_LIST = { items: [], total: 0, total_pages: 0 }
const TABS = [['story', '스토리 만들기'], ['code', '코드봇'], ['records', '스토리 기록'], ['mine', '내 기록']]

export default function App() {
  const [user, setUser] = useState(null)
  const [authReady, setAuthReady] = useState(false)
  const [authMode, setAuthMode] = useState(null)
  const [view, setView] = useState('story')
  const [prompt, setPrompt] = useState('')
  const [code, setCode] = useState('')
  const [story, setStory] = useState(null)
  const [completion, setCompletion] = useState(null)
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
  const isCode = view === 'code'
  const isRecords = view === 'records' || view === 'mine'
  useEffect(() => {
    api('/api/auth/me').then(setUser).catch(failure => {
      if (failure.status !== 401) setError('서버에 연결할 수 없습니다. 다시 접속해 주세요.')
    }).finally(() => setAuthReady(true))
  }, [])
  useEffect(() => {
    if (!user || !isRecords) return
    let active = true
    setListBusy(true)
    api(`/api/stories?page=${page}&mine=${view === 'mine'}`).then(data => {
      if (!active) return
      if (page > Math.max(1, data.total_pages)) setPage(Math.max(1, data.total_pages))
      else setListing(data)
    }).catch(failure => { if (active) handleFailure(failure) })
      .finally(() => { if (active) setListBusy(false) })
    return () => { active = false }
  }, [user?.id, view, page, revision])
  function handleFailure(failure) {
    setError(failure.message || '서버에 연결할 수 없습니다. 다시 시도해 주세요.')
    if (failure.status === 401) {
      setUser(null); setStory(null); setCompletion(null); setListing(EMPTY_LIST)
      setEditing(null); setDeleting(null); setAuthMode('login')
    }
  }
  function navigate(next) { setView(next); setPage(1); setError(''); setNotice('') }
  function signedIn(updated) {
    setUser(updated); setError(''); setNotice(''); setRevision(value => value + 1)
    if (story?.is_owner && user?.id === updated.id) setStory(previous => ({ ...previous, author: updated.nickname }))
  }
  async function logout() {
    try { await api('/api/auth/logout', { method: 'POST' }) }
    catch (failure) { if (failure.status !== 401) return handleFailure(failure) }
    setUser(null); setStory(null); setCompletion(null); setListing(EMPTY_LIST); navigate('story')
  }
  async function generate(event) {
    event.preventDefault()
    if (!user) return setAuthMode('login')
    if (busy) return
    const input = isCode ? code : prompt
    if (!input.trim()) return setError(isCode ? '코드를 입력해 주세요.' : '이야기의 시작을 입력해 주세요.')
    setBusy(true); setError(''); setNotice('')
    if (isCode) setCompletion(null)
    else setStory(null)
    try {
      const result = await api(isCode ? '/api/generate' : '/api/stories/generate', {
        method: 'POST', body: JSON.stringify(isCode ? { code } : { prompt }),
      })
      if (isCode) setCompletion(result.completion)
      else { setStory(result); setRevision(value => value + 1); setNotice('스토리가 공개 기록에 저장되었습니다.') }
    } catch (failure) { handleFailure(failure) }
    finally { setBusy(false) }
  }
  function updateVisibleStory(updated) {
    setListing(previous => ({ ...previous, items: previous.items.map(item => item.id === updated.id ? updated : item) }))
    setStory(previous => previous?.id === updated.id ? updated : previous)
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
  const cardActions = { onLike: toggleLike, onEdit: setEditing, onDelete: setDeleting }
  return <div className="site-shell">
    <header className="site-header"><div className="header-inner">
      <button className="brand" onClick={() => navigate('story')} disabled={busy} aria-label="스토리봇 홈"><span className="brand-symbol" aria-hidden="true">S</span><span>STORY<span className="brand-light">BOT</span></span></button>
      <span className="header-caption">당신의 한 줄에서 시작되는 이야기</span>
      <div className="header-account">{user ? <><span>{user.nickname}님</span><button onClick={logout} className="text-button" disabled={busy}>로그아웃</button></> : <button className="text-button" onClick={() => setAuthMode('login')}>로그인</button>}</div>
    </div></header>
    <main className="portal">
      <nav className="service-nav" aria-label="서비스">{TABS.map(([key, label]) => <button key={key} aria-current={view === key ? 'page' : undefined} className={view === key ? 'active' : ''} onClick={() => navigate(key)} disabled={busy}>{label}</button>)}</nav>
      <div className="portal-grid"><div className="main-column">
        {error && <p className="error" role="alert">{error}</p>}{notice && <p className="notice" role="status">{notice}</p>}
        {!isRecords ? <>
          <section className="card editor-card"><div className="section-heading"><div><span className="section-kicker">{isCode ? 'CODE BOT' : 'STORY BOT'}</span><h1>{isCode ? '코드의 다음 줄을 이어 보세요' : '어떤 이야기를 시작해 볼까요?'}</h1></div><span className="green-badge">최대 256토큰</span></div>
            <p className="section-description">{isCode ? '작성 중인 코드를 입력하면 뒷부분을 만들어 줍니다.' : '짧은 시작 문장을 입력하면 스토리봇이 다음 이야기를 이어 씁니다.'}</p>
            <form onSubmit={generate}><label className="field-label" htmlFor="bot-input">{isCode ? '작성 중인 코드' : '이야기의 시작'}</label>
              <textarea id="bot-input" className={isCode ? 'code-input' : 'story-input'} rows={6} value={isCode ? code : prompt} onChange={e => isCode ? setCode(e.target.value) : setPrompt(e.target.value)} placeholder={isCode ? 'def add(a, b):\n    ' : 'Once upon a time, a little fox found a door in the forest.'} spellCheck={false} disabled={busy} aria-describedby="input-help" />
              <div className="editor-bottom"><p id="input-help">{isCode ? '들여쓰기와 줄바꿈을 그대로 입력해 주세요.' : '영어로 시작하면 더 자연스러운 이야기를 만들 수 있어요.'}<br />토큰 수는 문자 수와 다릅니다.</p>
                <button className="primary-button" disabled={busy || !authReady}>{busy ? '생성하고 있습니다…' : !user ? '로그인하고 시작하기' : isCode ? '코드 이어 쓰기' : '이야기 이어 쓰기'}{!busy && <span aria-hidden="true"> →</span>}</button></div>
            </form>{!isCode && <p className="save-notice">생성한 스토리는 모든 회원이 보는 공개 기록에 자동으로 저장됩니다.</p>}
          </section>
          <section className="card output-card" aria-busy={busy} aria-labelledby="output-title"><div className="output-heading"><h2 id="output-title">{isCode ? '이어 쓴 코드' : '완성된 이야기'}</h2><span>최대 200토큰 생성</span></div>
            <div aria-live="polite">{busy ? <div className="empty-state"><span className="spark" aria-hidden="true">✦</span><p>다음 내용을 생각하고 있습니다…</p></div>
              : isCode ? completion === null ? <div className="empty-state"><span className="spark" aria-hidden="true">{'{ }'}</span><p>이어 쓴 코드가 여기에 표시됩니다.</p></div>
                : completion === '' ? <div className="empty-state"><p>생성된 뒷부분이 없습니다. 다시 시도해 주세요.</p></div> : <pre className="code-result"><code>{completion}</code></pre>
                : story ? <StoryCard story={story} {...cardActions} busy={pendingId !== null} /> : <div className="empty-state"><span className="spark" aria-hidden="true">✦</span><h3>아직 쓰이지 않은 다음 장</h3><p>시작 문장 하나로 새로운 이야기를 만나 보세요.</p></div>}
            </div>
          </section>
        </> : <section className="card records-card"><div className="section-heading"><div><span className="section-kicker">STORY ARCHIVE</span><h1>{view === 'mine' ? '내가 만든 이야기' : '함께 만든 이야기'}</h1></div>{user && <span className="record-count">총 {listing.total}개</span>}</div>
          <p className="section-description">{view === 'mine' ? '내 스토리를 다시 읽고, 고치고, 관리해 보세요.' : '다른 회원의 이야기를 읽고 마음에 드는 스토리에 좋아요를 눌러 보세요.'}</p>
          {!user ? <div className="empty-state"><p>로그인하면 스토리 기록을 볼 수 있습니다.</p><button className="primary-button" onClick={() => setAuthMode('login')}>로그인</button></div>
            : listBusy ? <p className="loading" role="status">기록을 불러오고 있습니다…</p>
              : listing.items.length === 0 ? <div className="empty-state"><span className="spark" aria-hidden="true">✦</span><p>아직 저장된 이야기가 없습니다.</p><button className="secondary-button" onClick={() => navigate('story')}>첫 이야기 만들기</button></div>
                : listing.items.map(item => <StoryCard key={item.id} story={item} {...cardActions} busy={pendingId !== null} />)}
          {user && listing.total_pages > 0 && <nav className="pagination" aria-label="기록 페이지"><button disabled={page <= 1 || listBusy} onClick={() => setPage(value => value - 1)}>이전</button><span aria-live="polite">{page} / {listing.total_pages}</span><button disabled={page >= listing.total_pages || listBusy} onClick={() => setPage(value => value + 1)}>다음</button></nav>}
        </section>}
      </div><aside className="sidebar">
        <section className="card account-card">{!authReady ? <p>로그인 상태를 확인하고 있습니다…</p> : user ? <>
          <div className="account-intro"><span className="large-avatar" aria-hidden="true">{user.nickname.slice(0, 1)}</span><div><strong>{user.nickname}님</strong><p>@{user.username}</p></div></div>
          <p className="account-greeting">오늘도 나만의 이야기를 만들어 보세요.</p><button className="secondary-button full-width" disabled={busy} onClick={() => navigate('mine')}>내 스토리 기록 보기</button>
          <div className="account-links"><button className="text-button" disabled={busy} onClick={() => setAuthMode('profile')}>회원정보 수정</button><span>·</span><button className="text-button" disabled={busy} onClick={logout}>로그아웃</button></div>
        </> : <><p>로그인하고 나만의 이야기를 모아 보세요.</p><button className="primary-button full-width" onClick={() => setAuthMode('login')}><b>STORYBOT</b> 로그인</button><button className="text-button join-link" onClick={() => setAuthMode('signup')}>회원가입</button></>}</section>
        <section className="card guide-card"><span className="section-kicker">작은 시작, 새로운 이야기</span><h2>스토리봇 이용 안내</h2><ol><li>이야기의 시작을 입력해 주세요.</li><li>스토리봇이 최대 200토큰을 이어 씁니다.</li><li>기록에서 다시 읽고 마음을 나눠 보세요.</li></ol><p>긴 입력은 생성 중 앞부분이 문맥에서 제외될 수 있습니다.</p></section>
        <p className="sidebar-note">스토리봇의 출력은 학습된 모델에 따라 달라집니다. 코드와 스토리는 화면에 표시하는 텍스트로만 사용됩니다.</p>
      </aside></div>
      <footer className="site-footer"><strong>STORYBOT</strong><span>한 줄의 시작, 나만의 이야기.</span></footer>
    </main>
    {authMode && <AuthDialog key={authMode} mode={authMode} user={user} onClose={() => setAuthMode(null)} onMode={setAuthMode} onSuccess={signedIn} />}
    {editing && <EditDialog story={editing} onClose={() => setEditing(null)} onSave={saveEdit} />}
    {deleting && <Modal title="스토리를 삭제할까요?" onClose={() => setDeleting(null)}><p className="delete-description">「{deleting.title}」와 연결된 좋아요가 삭제됩니다. 삭제한 스토리는 되돌릴 수 없습니다.</p><div className="confirm-actions"><button className="secondary-button" disabled={pendingId !== null} onClick={() => setDeleting(null)}>취소</button><button className="danger-button" disabled={pendingId !== null} onClick={confirmDelete}>{pendingId !== null ? '삭제 중…' : '삭제하기'}</button></div></Modal>}
  </div>
}
