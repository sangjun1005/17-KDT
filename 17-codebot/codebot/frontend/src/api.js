export async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: 'same-origin', ...options,
    headers: { 'Content-Type': 'application/json', ...options.headers },
  })
  if (response.status === 204) return null
  const data = await response.json()
  if (!response.ok) {
    const detail = data.detail
    const error = new Error(typeof detail === 'string' ? detail : detail?.message || '입력값을 확인해 주세요. 아이디는 영문·숫자·밑줄 3~30자, 비밀번호는 8~128자입니다.')
    error.status = response.status
    throw error
  }
  return data
}
