import { api } from './client'

export interface LoginPayload { email: string; password: string }
export interface RegisterPayload { email: string; password: string }
export interface AuthResponse { token: string; user: { id: string; email: string; hasCompanyInfo: boolean } }

export async function register(payload: RegisterPayload): Promise<AuthResponse> {
  const { data } = await api.post<AuthResponse>('/auth/register', payload)
  return data
}

export async function login(payload: LoginPayload): Promise<AuthResponse> {
  const { data } = await api.post<AuthResponse>('/auth/login', payload)
  return data
}
