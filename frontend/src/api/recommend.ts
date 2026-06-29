import { RecommendCard } from '../types'

// --- 더미 구현 (백엔드 GET /recommend 추가 전까지) ---
export async function fetchRecommend(): Promise<RecommendCard[]> {
  // const { data } = await api.get<RecommendCard[]>('/recommend')
  // return data

  await new Promise((r) => setTimeout(r, 600))
  return [
    {
      id: '1',
      title: '2025년 스마트 행정서비스 구축 사업',
      organization: '행정안전부',
      deadline: '2025-07-31',
      matchScore: 92,
      reason: '귀사의 IT서비스 업종 및 중소기업 인증이 입찰 자격 요건과 일치합니다.',
      item: {},
    },
    {
      id: '2',
      title: '공공 데이터 플랫폼 고도화 사업',
      organization: '한국지능정보사회진흥원',
      deadline: '2025-08-15',
      matchScore: 85,
      reason: '귀사의 소프트웨어 개발 역량과 ISO 인증이 우선 평가 항목에 해당합니다.',
      item: {},
    },
    {
      id: '3',
      title: '지역 중소기업 디지털 전환 지원 사업',
      organization: '중소벤처기업부',
      deadline: '2025-08-22',
      matchScore: 78,
      reason: '중소기업 자격 및 창업 연수 조건이 충족됩니다.',
      item: {},
    },
  ]
}
