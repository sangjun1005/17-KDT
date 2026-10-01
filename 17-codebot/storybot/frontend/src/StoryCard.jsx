export default function StoryCard({ story, onLike, onEdit, onDelete, busy }) {
  // 생성 직후처럼 내용이 시작 문장으로 시작하면 입력 부분과 생성 부분을 다른 색으로 표시
  const split = story.content.startsWith(story.prompt) && story.content !== story.prompt
  return <article className="story-card">
    <div className="story-meta"><span className="author-avatar" aria-hidden="true">{story.author.slice(0, 1)}</span>
      <span className="story-author">{story.author}</span>{story.is_owner && <span className="mine-badge">내 스토리</span>}
      <time dateTime={story.created_at}>{new Date(story.created_at).toLocaleString('ko-KR')}</time></div>
    <h3>{story.title}</h3>
    <p className="story-content">{split
      ? <><span className="prompt-part">{story.prompt}</span>{story.content.slice(story.prompt.length)}</>
      : story.content}</p>
    <div className="story-actions">
      <button className={`like-button ${story.liked ? 'liked' : ''}`} onClick={() => onLike(story)} disabled={busy} aria-pressed={story.liked}>
        <span aria-hidden="true">{story.liked ? '♥' : '♡'}</span> 좋아요 {story.like_count}</button>
      {story.is_owner && <div className="owner-actions">
        <button className="edit-button" onClick={() => onEdit(story)} disabled={busy}>수정</button>
        <button className="delete-button" onClick={() => onDelete(story)} disabled={busy}>삭제</button>
      </div>}
    </div>
  </article>
}
