import axios from 'axios'
import { api } from './client'
import { NaraNotice } from '../types'

export async function naraSearch(q: string, days: number, count: number): Promise<NaraNotice[]> {
  const { data } = await api.get('/nara/search', { params: { q, days, count }, timeout: 60_000 })
  return data.notices as NaraNotice[]
}

export async function naraIngest(item: Record<string, unknown>): Promise<{ 파일명: string }> {
  const { data } = await api.post('/nara/ingest', { item }, { timeout: 600_000 })
  return data
}

// 백엔드가 422로 응답하면 "첨부는 있지만 대화에 쓸 내용이 없음"(backend/app.py의
// nara_ingest, chunks=0)을 뜻한다 — 네트워크 오류 등 다른 실패와 구분해 안내 문구를 다르게 준다.
// PREFIX + LINK_TEXT를 이어 붙이면 전체 문구가 되고, bidNtceDtlUrl이 있으면 LINK_TEXT만
// 링크로 감싼다(없으면 그냥 이어붙인 텍스트로 표시).
export const NARA_NO_CONTENT_PREFIX = '이 공고는 첨부파일에서 대화에 필요한 내용을 찾지 못했어요. 원문은 '
export const NARA_NO_CONTENT_LINK_TEXT = '나라장터에서 직접 확인해주세요.'
export const NARA_NO_CONTENT_MESSAGE = NARA_NO_CONTENT_PREFIX + NARA_NO_CONTENT_LINK_TEXT

export function isNaraNoContentError(err: unknown): boolean {
  return axios.isAxiosError(err) && err.response?.status === 422
}
