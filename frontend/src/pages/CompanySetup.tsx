import { useState, FormEvent, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getCompany, saveCompany, updateCompany } from '../api/company'
import { markCompanyInfoDone } from '../api/auth'
import { useStore } from '../store/useStore'
import { CompanyInfo } from '../types'

const CERT_OPTIONS = [
  'ISO 9001', 'ISO 14001', 'ISO 27001', 'ISMS-P',
  '벤처기업', '기술혁신기업(Inno-Biz)', '경영혁신기업(Main-Biz)',
  '여성기업', '사회적기업', '장애인기업', '녹색인증',
]

const SIZE_OPTIONS = ['대기업', '중견기업', '중소기업', '소기업', '스타트업']
const INDUSTRY_OPTIONS = [
  'IT서비스/소프트웨어', '시스템통합(SI)', '데이터/AI',
  '건설/엔지니어링', '컨설팅', '교육/훈련', '의료/바이오',
  '제조', '유통/물류', '기타',
]

const EMPTY: CompanyInfo = {
  industry: '', size: '', foundingYears: 0, annualRevenue: '',
  headcount: 0, certifications: [], additionalInfo: '',
}

export default function CompanySetup() {
  const navigate = useNavigate()
  const user = useStore((s) => s.user)
  const setHasCompanyInfo = useStore((s) => s.setHasCompanyInfo)
  const [form, setForm] = useState<CompanyInfo>(EMPTY)
  const [isEdit, setIsEdit] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    getCompany().then((data) => {
      if (data) { setForm(data); setIsEdit(true) }
    })
  }, [])

  function toggleCert(cert: string) {
    setForm((f) => ({
      ...f,
      certifications: f.certifications.includes(cert)
        ? f.certifications.filter((c) => c !== cert)
        : [...f.certifications, cert],
    }))
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!form.industry || !form.size) { setError('업종과 기업 규모를 선택해주세요.'); return }
    setError('')
    setLoading(true)
    try {
      if (isEdit) { await updateCompany(form) } else { await saveCompany(form) }
      if (user?.email) await markCompanyInfoDone(user.email)
      setHasCompanyInfo(true)
      navigate('/')
    } catch {
      setError('저장에 실패했습니다. 다시 시도해주세요.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex items-start justify-center px-4 py-12">
      <div className="card w-full max-w-2xl p-8">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 rounded-lg bg-primary-600 flex items-center justify-center text-white font-bold text-sm">
              1
            </div>
            <h1 className="text-xl font-bold text-gray-900">
              {isEdit ? '회사 정보 수정' : '회사 정보 입력'}
            </h1>
          </div>
          <p className="text-sm text-gray-500 ml-11">
            입찰 적격성 판정 및 공고 추천에 사용되는 회사 정보를 입력해주세요.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* 업종 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">업종 *</label>
            <select
              className="input-field"
              value={form.industry}
              onChange={(e) => setForm((f) => ({ ...f, industry: e.target.value }))}
            >
              <option value="">선택하세요</option>
              {INDUSTRY_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </div>

          {/* 기업 규모 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">기업 규모 *</label>
            <div className="flex flex-wrap gap-2">
              {SIZE_OPTIONS.map((s) => (
                <button
                  key={s} type="button"
                  onClick={() => setForm((f) => ({ ...f, size: s }))}
                  className={`px-4 py-2 rounded-lg text-sm font-medium border transition-colors ${
                    form.size === s
                      ? 'bg-primary-600 text-white border-primary-600'
                      : 'bg-white text-gray-600 border-gray-200 hover:border-primary-400'
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          {/* 창업 연수 / 직원 수 */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1.5">창업 연수 (년)</label>
              <input
                type="number" min={0} max={200}
                className="input-field"
                placeholder="예: 5"
                value={form.foundingYears || ''}
                onChange={(e) => setForm((f) => ({ ...f, foundingYears: Number(e.target.value) }))}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1.5">임직원 수 (명)</label>
              <input
                type="number" min={0}
                className="input-field"
                placeholder="예: 30"
                value={form.headcount || ''}
                onChange={(e) => setForm((f) => ({ ...f, headcount: Number(e.target.value) }))}
              />
            </div>
          </div>

          {/* 매출액 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">연 매출액</label>
            <input
              type="text"
              className="input-field"
              placeholder="예: 30억, 5억 미만"
              value={form.annualRevenue}
              onChange={(e) => setForm((f) => ({ ...f, annualRevenue: e.target.value }))}
            />
          </div>

          {/* 인증 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">보유 인증</label>
            <div className="flex flex-wrap gap-2">
              {CERT_OPTIONS.map((c) => (
                <button
                  key={c} type="button"
                  onClick={() => toggleCert(c)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                    form.certifications.includes(c)
                      ? 'bg-primary-600 text-white border-primary-600'
                      : 'bg-white text-gray-600 border-gray-200 hover:border-primary-400'
                  }`}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>

          {/* 추가 정보 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">추가 정보 (선택)</label>
            <textarea
              className="input-field resize-none"
              rows={3}
              placeholder="주력 사업, 수행 실적, 특수 역량 등 입찰에 도움이 될 추가 정보를 입력하세요."
              value={form.additionalInfo}
              onChange={(e) => setForm((f) => ({ ...f, additionalInfo: e.target.value }))}
            />
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3">
              {error}
            </div>
          )}

          <div className="flex gap-3 pt-2">
            {isEdit && (
              <button type="button" className="btn-secondary flex-1" onClick={() => navigate('/')}>
                취소
              </button>
            )}
            <button type="submit" className="btn-primary flex-1 py-3" disabled={loading}>
              {loading ? '저장 중...' : isEdit ? '수정 완료' : '저장하고 시작하기'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
