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

async function fetchNotices() {
  // 스케줄러 캐시 우선 시도, 비어있거나 실패하면 직접 검색으로 fallback
  try {
    const { data } = await api.get<{ notices: unknown[] }>('/nara/recommend-cache')
    if (data.notices?.length) return data.notices as import('../types').NaraNotice[]
  } catch { /* ignore */ }
  return naraSearch('', 7)
}

export async function fetchRecommend(): Promise<RecommendCard[]> {
  // 1. 나라장터 최근 7일 공고 목록 (캐시 우선, fallback: 직접 검색)
  const notices = await fetchNotices()
  if (!notices.length) return []

  // 2. localStorage 회사 정보
  let companyInfo: CompanyInfo | null = null
  try {
    companyInfo = JSON.parse(localStorage.getItem('company_info') ?? 'null')
  } catch { /* ignore */ }

  const profile = buildProfile(companyInfo)
  const today = new Date()
  today.setHours(0, 0, 0, 0)

  const filteredNotices = notices.filter((n) => {
    if (!(n.item.ntceSpecDocUrl1 as string)) return false
    const raw = (n.item.bidClseDt as string) ?? ''
    const digits = raw.replace(/\D/g, '')
    if (digits.length < 8) return false
    const closeDate = new Date(`${digits.slice(0, 4)}-${digits.slice(4, 6)}-${digits.slice(6, 8)}`)
    return closeDate >= today
  })
  const bids = filteredNotices.map((n) => n.item)

  // bidNtceNo → 원본 notice.item 역참조용 (인덱싱 시 _score 없는 원본을 전달하기 위함)
  const noticeItemByNo = new Map<string, Record<string, unknown>>(
    filteredNotices.flatMap((n) => {
      const no = n.item.bidNtceNo as string | undefined
      return no ? [[no, n.item]] : []
    }),
  )

  // 3. 점수화 — top_n:3 이므로 최대 3건만 반환됨
  const { data } = await api.post<{ results: Record<string, unknown>[] }>(
    '/recommend',
    { profile, bids },
  )
  const results = data.results ?? []

  return results.map((r, i) => {
    const no = r.bidNtceNo as string | undefined
    // card.item은 _score가 없는 원본 나라장터 API 응답 — /nara/ingest payload로 직접 사용
    const noticeItem = (no ? noticeItemByNo.get(no) : undefined) ?? r
    return {
      id: no ?? String(i),
      title: (r.bidNtceNm as string) ?? '(사업명 없음)',
      organization: ((r.dmndInsttNm ?? r.ntceInsttNm) as string) ?? '',
      deadline: parseDeadline(r),
      matchScore: (r._score as number) ?? 0,
      reason: makeReason((r._score as number) ?? 0, r),
      item: noticeItem,
    }
  })
}
