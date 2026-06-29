import { api } from './client'

export interface LoginPayload { email: string; password: string }
export interface RegisterPayload { email: string; password: string }
export interface AuthResponse { token: string; user: { id: string; email: string; hasCompanyInfo: boolean } }

// --- 더미 구현 (백엔드 /auth/* 엔드포인트 추가 전까지) ---
const DUMMY_USERS_KEY = 'dummy_users'

function getUsers(): Record<string, { password: string; id: string; hasCompanyInfo: boolean }> {
  try { return JSON.parse(localStorage.getItem(DUMMY_USERS_KEY) || '{}') }
  catch { return {} }
}

function saveUsers(u: ReturnType<typeof getUsers>) {
  localStorage.setItem(DUMMY_USERS_KEY, JSON.stringify(u))
}

export async function register(payload: RegisterPayload): Promise<AuthResponse> {
  // 실제 엔드포인트가 생기면 아래 주석을 해제하고 더미 코드를 제거하세요.
  // const { data } = await api.post<AuthResponse>('/auth/register', payload)
  // return data

  await new Promise((r) => setTimeout(r, 400))
  const users = getUsers()
  if (users[payload.email]) throw new Error('이미 사용 중인 이메일입니다.')
  const id = Math.random().toString(36).slice(2)
  users[payload.email] = { password: payload.password, id, hasCompanyInfo: false }
  saveUsers(users)
  const token = btoa(`${id}:${payload.email}`)
  return { token, user: { id, email: payload.email, hasCompanyInfo: false } }
}

export async function login(payload: LoginPayload): Promise<AuthResponse> {
  // const { data } = await api.post<AuthResponse>('/auth/login', payload)
  // return data

  await new Promise((r) => setTimeout(r, 400))
  const users = getUsers()
  const u = users[payload.email]
  if (!u || u.password !== payload.password) throw new Error('이메일 또는 비밀번호가 올바르지 않습니다.')
  const token = btoa(`${u.id}:${payload.email}`)
  return { token, user: { id: u.id, email: payload.email, hasCompanyInfo: u.hasCompanyInfo } }
}

export async function markCompanyInfoDone(email: string) {
  const users = getUsers()
  if (users[email]) { users[email].hasCompanyInfo = true; saveUsers(users) }
}
