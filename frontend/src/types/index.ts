export interface User {
  id: string
  email: string
  hasCompanyInfo: boolean
}

export interface CompanyInfo {
  industry: string
  size: string       // 대기업 / 중견기업 / 중소기업 / 소기업 / 스타트업
  foundingYears: number
  annualRevenue: string
  headcount: number
  certifications: string[]
  additionalInfo: string
}

export interface Source {
  파일명?: string
  기관?: string
  금액?: number | null
  마감일?: string
}

export interface ChatResponse {
  answer: string
  refused?: boolean
  sources?: Source[]
}

export interface InfographicResponse {
  html?: string
  parse_error?: string
  sources?: Source[]
}

export type TurnKind = 'chat' | 'infographic'

export interface Turn {
  kind: TurnKind
  question: string
  data: ChatResponse | InfographicResponse
}

export interface Document {
  파일명: string
  사업명?: string
  기관?: string
  금액?: number | null
  마감일?: string
}

export interface Agent {
  id: string
  label: string
  description: string
  implemented: boolean
  endpoint?: string
  prompt?: string
  needs_company_info?: boolean
}

export interface NaraNotice {
  사업명?: string
  기관?: string
  마감일?: string
  금액?: number | null
  item: Record<string, unknown>
}

export interface ConfigCatalog {
  llm?: { presets: Preset[]; current: string }
  embedding?: { presets: Preset[]; current: string }
  chunk?: { presets: Preset[]; current: string }
  parser_hwp?: { presets: Preset[]; current: string }
  parser_pdf?: { presets: Preset[]; current: string }
}

export interface Preset {
  key: string
  label: string
}

export interface Config {
  catalog: ConfigCatalog
  settings: {
    retriever?: { k: number; use_bm25: boolean; use_reranker: boolean }
    llm?: { temperature: number }
  }
}

export interface RecommendCard {
  id: string
  title: string
  organization: string
  deadline: string
  budget: number | null
  contractMethod: string | null
  region: string | null
  matchScore: number
  reasonTags: string[]
  item: Record<string, unknown>
}
