import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { fetchConfig, applyConfig } from '../api/config'
import { Preset } from '../types'
import { useNavigate } from 'react-router-dom'

function PresetSelect({ label, presets, value, onChange, helpReindex }: {
  label: string
  presets: Preset[]
  value: string
  onChange: (v: string) => void
  helpReindex?: boolean
}) {
  if (!presets.length) return null
  return (
    <div>
      <label className="block text-xs font-medium text-gray-600 mb-1">
        {label} {helpReindex && <span className="text-gray-400">(변경 시 재인덱싱)</span>}
      </label>
      <select className="input-field text-xs" value={value} onChange={(e) => onChange(e.target.value)}>
        {presets.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
      </select>
    </div>
  )
}

export default function Settings() {
  const navigate = useNavigate()
  const { data: config, isLoading } = useQuery({ queryKey: ['config'], queryFn: fetchConfig })
  const [llm, setLlm] = useState('')
  const [emb, setEmb] = useState('')
  const [chunk, setChunk] = useState('')
  const [hwp, setHwp] = useState('')
  const [pdf, setPdf] = useState('')
  const [k, setK] = useState(4)
  const [bm25, setBm25] = useState(false)
  const [reranker, setReranker] = useState(false)
  const [temp, setTemp] = useState(0)
  const [initialized, setInitialized] = useState(false)
  const [success, setSuccess] = useState('')

  if (config && !initialized) {
    setLlm(config.catalog.llm?.current || '')
    setEmb(config.catalog.embedding?.current || '')
    setChunk(config.catalog.chunk?.current || '')
    setHwp(config.catalog.parser_hwp?.current || '')
    setPdf(config.catalog.parser_pdf?.current || '')
    setK(config.settings.retriever?.k ?? 4)
    setBm25(config.settings.retriever?.use_bm25 ?? false)
    setReranker(config.settings.retriever?.use_reranker ?? false)
    setTemp(config.settings.llm?.temperature ?? 0)
    setInitialized(true)
  }

  const mutation = useMutation({
    mutationFn: applyConfig,
    onSuccess: (data) => {
      setSuccess(data.reindexed ? '재인덱싱 완료' : '설정 적용 완료 (기존 인덱스 재사용)')
      setTimeout(() => setSuccess(''), 3000)
    },
  })

  function handleApply() {
    mutation.mutate({
      llm_preset: llm, embedding_preset: emb, chunk_preset: chunk,
      parser_hwp_preset: hwp, parser_pdf_preset: pdf,
      retriever_k: k, retriever_use_bm25: bm25,
      retriever_use_reranker: reranker, llm_temperature: temp,
    })
  }

  if (isLoading) return <p className="text-xs text-gray-400 py-2">설정 불러오는 중...</p>
  if (!config) return <p className="text-xs text-red-400 py-2">설정을 불러올 수 없습니다.</p>

  const c = config.catalog

  return (
    <div className="space-y-4">
      <div>
        <button
          onClick={() => navigate('/company-setup')}
          className="w-full flex items-center gap-2 py-1.5 text-xs text-gray-600 hover:text-primary-600 transition-colors"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
          </svg>
          회사 정보 수정
        </button>
      </div>

      <div className="border-t border-gray-100 pt-3">
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">RAG 설정</p>
        <div className="space-y-3">
          <PresetSelect label="LLM 모델" presets={c.llm?.presets || []} value={llm} onChange={setLlm} />
          <PresetSelect label="임베딩" presets={c.embedding?.presets || []} value={emb} onChange={setEmb} helpReindex />
          <PresetSelect label="청킹" presets={c.chunk?.presets || []} value={chunk} onChange={setChunk} helpReindex />
          <PresetSelect label="HWP 파서" presets={c.parser_hwp?.presets || []} value={hwp} onChange={setHwp} helpReindex />
          <PresetSelect label="PDF 파서" presets={c.parser_pdf?.presets || []} value={pdf} onChange={setPdf} helpReindex />

          <div>
            <label className="flex justify-between text-xs font-medium text-gray-600 mb-1">
              <span>검색 문서 수 (k)</span>
              <span className="text-primary-600">{k}</span>
            </label>
            <input type="range" min={1} max={10} value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="w-full accent-primary-600"
            />
          </div>

          <div>
            <label className="flex justify-between text-xs font-medium text-gray-600 mb-1">
              <span>Temperature</span>
              <span className="text-primary-600">{temp.toFixed(1)}</span>
            </label>
            <input type="range" min={0} max={1} step={0.1} value={temp}
              onChange={(e) => setTemp(Number(e.target.value))}
              className="w-full accent-primary-600"
            />
          </div>

          <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
            <input type="checkbox" checked={bm25} onChange={(e) => setBm25(e.target.checked)}
              className="accent-primary-600" />
            BM25 하이브리드 검색
          </label>
          <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
            <input type="checkbox" checked={reranker} onChange={(e) => setReranker(e.target.checked)}
              className="accent-primary-600" />
            리랭커 (bge-reranker-v2-m3)
          </label>
        </div>
      </div>

      {success && (
        <div className="bg-green-50 border border-green-200 text-green-700 text-xs rounded-lg px-3 py-2">
          {success}
        </div>
      )}
      {mutation.isError && (
        <div className="bg-red-50 border border-red-200 text-red-600 text-xs rounded-lg px-3 py-2">
          설정 적용에 실패했습니다.
        </div>
      )}

      <button
        className="btn-primary w-full text-xs py-2"
        onClick={handleApply}
        disabled={mutation.isPending}
      >
        {mutation.isPending ? '적용 중...' : '설정 적용'}
      </button>
    </div>
  )
}
