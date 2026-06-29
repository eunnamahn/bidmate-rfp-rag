import { api } from './client'
import { CompanyInfo } from '../types'

export async function getCompany(): Promise<CompanyInfo | null> {
  try {
    const { data } = await api.get<CompanyInfo>('/company')
    return data
  } catch (err: any) {
    if (err?.response?.status === 404) return null
    throw err
  }
}

export async function saveCompany(info: CompanyInfo): Promise<void> {
  await api.post('/company', info)
}

export async function updateCompany(info: CompanyInfo): Promise<void> {
  await api.put('/company', info)
}
