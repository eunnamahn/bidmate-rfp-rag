import { memo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import { Turn, ChatResponse, InfographicResponse, Source } from '../types'
import MermaidDiagram from './MermaidDiagram'
import { fileViewUrl, fileDownloadUrl } from '../api/documents'

const MERMAID_RE = /```mermaid\s*([\s\S]*?)```/

const RISK_BADGE_MAP: [RegExp, string][] = [
  [/\[높음\]/g, '<span style="background:#fee2e2;color:#b91c1c;padding:2px 6px;border-radius:4px;font-weight:bold">[높음]</span>'],
  [/\[중간\]/g, '<span style="background:#ffedd5;color:#c2410c;padding:2px 6px;border-radius:4px;font-weight:bold">[중간]</span>'],
  [/\[낮음\]/g, '<span style="background:#dcfce7;color:#15803d;padding:2px 6px;border-radius:4px;font-weight:bold">[낮음]</span>'],
]

function injectRiskBadges(text: string): string {
  return RISK_BADGE_MAP.reduce((s, [re, html]) => s.replace(re, html), text)
}

function SourcesPanel({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false)
  if (!sources.length) return null
  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen((o) => !o)}
        className="text-xs text-gray-400 hover:text-primary-600 flex items-center gap-1 transition-colors"
      >
        <svg className={`w-3.5 h-3.5 transition-transform ${open ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        근거 출처 {sources.length}건
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {sources.map((s, i) => {
            const name = s.파일명 || '(파일명 없음)'
            return (
              <div key={i} className="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
                <p className="font-medium text-gray-700">{i + 1}. {name}</p>
                <p className="mt-0.5">
                  기관: {s.기관 || '미상'} · 금액: {s.금액 != null ? `${Number(s.금액).toLocaleString()}원` : '미기재'} · 마감: {s.마감일 || '미상'}
                </p>
                {s.파일명 && (
                  <div className="mt-1.5 flex gap-3">
                    <a href={fileViewUrl(s.파일명)} target="_blank" rel="noreferrer"
                      className="text-primary-600 hover:underline">파일 보기</a>
                    <a href={fileDownloadUrl(s.파일명)} target="_blank" rel="noreferrer"
                      className="text-primary-600 hover:underline">다운로드</a>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// memo로 감싸 새 메시지 추가 시 ActionBar prop 변경으로 인한 부모 리렌더가
// AnswerContent까지 전파되지 않게 한다. data가 동일 참조라면 MermaidDiagram도 리렌더되지 않는다.
const AnswerContent = memo(function AnswerContent({ data }: { data: ChatResponse }) {
  const answer = injectRiskBadges(data.answer || '')
  const m = MERMAID_RE.exec(answer)
  if (m) {
    const before = answer.slice(0, m.index).trim()
    const code = m[1].trim()
    const after = answer.slice(m.index + m[0].length).trim()
    return (
      <>
        {before && <div className="prose-chat"><ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>{before}</ReactMarkdown></div>}
        <MermaidDiagram code={code} />
        {after && <div className="prose-chat"><ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>{after}</ReactMarkdown></div>}
        <SourcesPanel sources={data.sources || []} />
      </>
    )
  }
  return (
    <>
      <div className="prose-chat"><ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>{answer}</ReactMarkdown></div>
      <SourcesPanel sources={data.sources || []} />
    </>
  )
})

// memo로 감싸 부모 리렌더(새 메시지 추가 등) 시 srcDoc prop이 재평가되지 않게 한다.
// srcDoc이 변경되면 iframe이 재로드되어 크기가 초기화되므로, 동일한 turn.data를 받는 한
// 컴포넌트 자체를 렌더링하지 않아 iframe을 그대로 유지한다.
const InfographicContent = memo(function InfographicContent({ data }: { data: InfographicResponse }) {
  if (data.parse_error) {
    return (
      <div>
        <p className="text-amber-700 text-sm mb-2">인포그래픽 정보 추출에 실패했습니다.</p>
        <pre className="text-xs bg-gray-100 rounded p-2 overflow-auto">{data.parse_error}</pre>
        <SourcesPanel sources={data.sources || []} />
      </div>
    )
  }
  if (data.html) {
    return (
      <>
        <div style={{ height: '760px', minHeight: '760px', width: '100%', flexShrink: 0, overflow: 'hidden' }}>
          <iframe
            srcDoc={data.html}
            className="rounded-lg border border-gray-100"
            style={{ height: '760px', minHeight: '760px', width: '100%', display: 'block', flexShrink: 0 }}
            sandbox="allow-scripts"
            title="인포그래픽"
          />
        </div>
        <SourcesPanel sources={data.sources || []} />
      </>
    )
  }
  return <p className="text-sm text-gray-500">인포그래픽을 생성하지 못했습니다.</p>
})

interface Props {
  turn: Turn
  actionBar?: React.ReactNode
}

export default function MessageBubble({ turn, actionBar }: Props) {
  return (
    <div className="space-y-1">
      {/* User bubble */}
      <div className="flex justify-end">
        <div className="max-w-[75%] bg-primary-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm">
          {turn.question}
        </div>
      </div>

      {/* AI bubble */}
      <div className="flex justify-start">
        <div className="flex gap-2.5 max-w-[85%]">
          <div className="flex-shrink-0 w-7 h-7 rounded-full bg-primary-100 flex items-center justify-center mt-0.5">
            <svg className="w-4 h-4 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm text-sm flex-1">
            {(turn.data as ChatResponse).refused && (
              <div className="flex items-start gap-2 text-amber-700 bg-amber-50 rounded-lg px-3 py-2 mb-3 text-xs">
                <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                문서에서 근거를 찾지 못했습니다.
              </div>
            )}
            {turn.kind === 'infographic'
              ? <InfographicContent data={turn.data as InfographicResponse} />
              : <AnswerContent data={turn.data as ChatResponse} />
            }
            {actionBar && <div className="mt-3 pt-3 border-t border-gray-100">{actionBar}</div>}
          </div>
        </div>
      </div>
    </div>
  )
}
