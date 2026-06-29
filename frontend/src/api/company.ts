import { CompanyInfo } from '../types'

const KEY = 'company_info'

// --- 더미 구현 (백엔드 /company 엔드포인트 추가 전까지) ---
export async function getCompany(): Promise<CompanyInfo | null> {
  // const { data } = await api.get<CompanyInfo>('/company')
  // return data
  try { return JSON.parse(localStorage.getItem(KEY) || 'null') }
  catch { return null }
}

export async function saveCompany(info: CompanyInfo): Promise<void> {
  // await api.post('/company', info)
  await new Promise((r) => setTimeout(r, 300))
  localStorage.setItem(KEY, JSON.stringify(info))
}

export async function updateCompany(info: CompanyInfo): Promise<void> {
  // await api.put('/company', info)
  await new Promise((r) => setTimeout(r, 300))
  localStorage.setItem(KEY, JSON.stringify(info))
}
