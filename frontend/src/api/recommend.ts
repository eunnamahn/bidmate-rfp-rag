import { naraSearch } from './nara'
import { api } from './client'
import { RecommendCard, CompanyInfo } from '../types'

// "30억", "5천만원", "1억5천만" 등 → 원 단위 정수
function parseRevenue(s: string): number {
  if (!s) return 0
  let total = 0
  const uk = s.match(/(\d+(?:\.\d+)?)억/)
  const chun = s.match(/(\d+(?:\.\d+)?)천만/)
  const man = s.match(/(\d+(?:\.\d+)?)만/)
  if (uk) total += parseFloat(uk[1]) * 1_0000_0000
  if (chun) total += parseFloat(chun[1]) * 1000_0000
  else if (!uk && man) total += parseFloat(man[1]) * 10000
  return Math.floor(total)
}

function buildProfile(info: CompanyInfo | null) {
  const rev = parseRevenue(info?.annualRevenue ?? '')
  return {
    mid_category: info?.industry ?? null,
    budget_min: null as number | null,
    budget_max: rev > 0 ? rev * 3 : (null as number | null),
    bid_methods: [] as string[],
    min_score: 30,
    top_n: 3,
  }
}

function parseDeadline(item: Record<string, unknown>): string {
  const raw = ((item.opengDt ?? item.bidClseDt) as string) ?? ''
  const digits = raw.replace(/\D/g, '')
  if (digits.length >= 8)
    return `${digits.slice(0, 4)}-${digits.slice(4, 6)}-${digits.slice(6, 8)}`
  return raw
}

function makeReason(score: number, item: Record<string, unknown>): string {
  const mid = (item.pubPrcrmntMidClsfcNm as string) ?? ''
  const method = (item.bidMethdNm as string) ?? ''
  if (score >= 70)
    return `업종 분류${mid ? ` (${mid})` : ''} 및 예산 범위가 귀사 조건과 높게 일치합니다.`
  if (score >= 50)
    return `주요 입찰 조건${method ? ` (${method})` : ''}이 귀사 프로필과 부분 일치합니다.`
  return '기본 입찰 자격을 충족하며 마감 여유가 있는 공고입니다.'
}

export async function fetchRecommend(): Promise<RecommendCard[]> {
  // 1. 나라장터 최근 7일 공고 목록 (메타데이터만, 다운로드 없음)
  const notices = await naraSearch('', 7)
  if (!notices.length) return []

  // 2. localStorage 회사 정보
  let companyInfo: CompanyInfo | null = null
  try {
    companyInfo = JSON.parse(localStorage.getItem('company_info') ?? 'null')
  } catch { /* ignore */ }

  const profile = buildProfile(companyInfo)
  const bids = notices.map((n) => n.item)

  // 3. 점수화 — top_n:3 이므로 최대 3건만 반환됨
  const { data } = await api.post<{ results: Record<string, unknown>[] }>(
    '/recommend',
    { profile, bids },
  )
  const results = data.results ?? []

  return results.map((r, i) => ({
    id: (r.bidNtceNo as string) ?? String(i),
    title: (r.bidNtceNm as string) ?? '(사업명 없음)',
    organization: ((r.dmndInsttNm ?? r.ntceInsttNm) as string) ?? '',
    deadline: parseDeadline(r),
    matchScore: (r._score as number) ?? 0,
    reason: makeReason((r._score as number) ?? 0, r),
    item: r,
  }))
}
