export default function StoryCard({ story, onLike, onEdit, onDelete, busy }) {
  return <article className="story-card">
    <div className="story-meta"><span className="author-avatar" aria-hidden="true">{story.author.slice(0, 1)}</span>
      <span className="story-author">{story.author}</span>{story.is_owner && <span className="mine-badge">내 스토리</span>}
      <time dateTime={story.created_at}>{new Date(story.created_at).toLocaleString('ko-KR')}</time></div>
    <h3>{story.title}</h3><p className="story-content">{story.content}</p>
    <div className="story-actions"><button className={`like-button ${story.liked ? 'liked' : ''}`} onClick={() => onLike(story)} disabled={busy} aria-pressed={story.liked} aria-label={`${story.liked ? '좋아요 취소' : '좋아요'} · ${story.title}`}>
      <span aria-hidden="true">{story.liked ? '♥' : '♡'}</span> 좋아요 {story.like_count}</button>
      {story.is_owner && <div className="owner-actions"><button className="text-button" onClick={() => onEdit(story)} disabled={busy}>수정</button><button className="text-button danger-text" onClick={() => onDelete(story)} disabled={busy}>삭제</button></div>}
    </div>
  </article>
}
