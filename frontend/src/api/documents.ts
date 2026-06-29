import { api, BACKEND_URL } from './client'
import { Document } from '../types'

export async function fetchDocuments(): Promise<Document[]> {
  const { data } = await api.get('/documents')
  return data.documents as Document[]
}

export async function uploadFile(file: File): Promise<{ 파일명: string; chunks: number }> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 600_000,
  })
  return data
}

export function fileViewUrl(filename: string) {
  return `${BACKEND_URL}/file?name=${encodeURIComponent(filename)}`
}

export function fileDownloadUrl(filename: string) {
  return `${BACKEND_URL}/file?name=${encodeURIComponent(filename)}&download=1`
}
