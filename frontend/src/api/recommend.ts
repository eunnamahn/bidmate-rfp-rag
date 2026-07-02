import { naraSearch } from './nara'
import { getCompany } from './company'
import { api } from './client'
import { RecommendCard, CompanyInfo } from '../types'

// 개수 선택 UI 없이, 프로필 기준을 넘긴 후보는 가능한 한 많이 받아 프론트 무한 스크롤로 공개한다.
const TOP_N = 200

// 근거 태그 키(백엔드 recommender.score_breakdown의 REASON_*) → 카드에 표시할 짧은 문구.
const REASON_LABELS: Record<string, string> = {
  category: '업종 일치',
  budget: '예산 규모 적합',
  method: '선호 입찰방식 일치',
  deadline: '마감 여유 충분',
}

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
    top_n: TOP_N,
  }
}

// 원본 나라장터 item에서 예산(추정가격/배정예산)을 원 단위 정수로 추출한다.
function parseBudget(item: Record<string, unknown>): number | null {
  const raw = (item.presmptPrce ?? item.asignBdgtAmt) as string | number | undefined
  if (raw === undefined || raw === null || raw === '') return null
  const n = Number(String(raw).replace(/[^\d.-]/g, ''))
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : null
}

function parseDeadline(item: Record<string, unknown>): string {
  const raw = ((item.opengDt ?? item.bidClseDt) as string) ?? ''
  const digits = raw.replace(/\D/g, '')
  if (digits.length >= 8)
    return `${digits.slice(0, 4)}-${digits.slice(4, 6)}-${digits.slice(6, 8)}`
  return raw
}

function makeReasonTags(reasons: unknown): string[] {
  if (!Array.isArray(reasons)) return []
  return reasons
    .map((r) => REASON_LABELS[r as string])
    .filter((label): label is string => !!label)
    .slice(0, 2)
}

// 점수화 전 후보 풀 — 무한 스크롤로 스크롤할 만큼 넉넉히 가져온다.
const CANDIDATE_POOL_SIZE = 100

async function fetchNotices() {
  // 스케줄러 캐시 우선 시도, 비어있거나 실패하면 직접 검색으로 fallback
  try {
    const { data } = await api.get<{ notices: unknown[] }>('/nara/recommend-cache')
    if (data.notices?.length) return data.notices as import('../types').NaraNotice[]
  } catch { /* ignore */ }
  return naraSearch('', 7, CANDIDATE_POOL_SIZE)
}

export async function fetchRecommend(): Promise<RecommendCard[]> {
  // 1. 나라장터 최근 7일 공고 목록 (캐시 우선, fallback: 직접 검색)
  const notices = await fetchNotices()
  if (!notices.length) return []

  // 2. 실제 회사 정보 — 백엔드 /company에서 조회 (localStorage에는 저장되지 않으므로 여기서 읽으면 안 됨)
  let companyInfo: CompanyInfo | null = null
  try {
    companyInfo = await getCompany()
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

  // 3. 점수화 — min_score 이상인 후보 전체를 점수 내림차순으로 받는다(개수 제한은 UI가 아닌 스크롤로 처리).
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
      budget: parseBudget(r),
      contractMethod: (r.cntrctCnclsMthdNm as string) || null,
      region: (r.rgnLmtBidLocplcJdgmBssNm as string) || null,
      matchScore: (r._score as number) ?? 0,
      reasonTags: makeReasonTags(r._reasons),
      item: noticeItem,
    }
  })
}
