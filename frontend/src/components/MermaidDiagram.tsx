import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import mermaid from 'mermaid'

mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose' })

let counter = 0

function sanitizeMermaid(code: string): string {
  const makeReplacer = (open: string, close: string) => {
    const pat = new RegExp(
      `\\${open}([^\\${open}\\${close}]*)\\${close}`,
      'g',
    )
    return (s: string) =>
      s.replace(pat, (_m, inner: string) => {
        const trimmed = inner.trim()
        if (!trimmed || (trimmed.startsWith('"') && trimmed.endsWith('"'))) return _m
        return `${open}"${trimmed.replace(/"/g, "'")}"${close}`
      })
  }
  let result = makeReplacer('[', ']')(code)
  result = makeReplacer('{', '}')(result)
  return result
}

interface SvgData {
  raw: string
  w: number
  h: number
}

function MermaidModal({ svgData, onClose }: { svgData: SvgData; onClose: () => void }) {
  const innerRef = useRef<HTMLDivElement>(null)
  const viewportRef = useRef<HTMLDivElement>(null)
  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const dragStart = useRef<{ mx: number; my: number; ox: number; oy: number } | null>(null)

  // SVG 초기 렌더 (0.8x = 현재 0.3x의 2배)
  useEffect(() => {
    if (!innerRef.current) return
    innerRef.current.innerHTML = svgData.raw
    const el = innerRef.current.querySelector('svg')
    if (!el) return
    el.removeAttribute('style')
    if (svgData.w > 0 && svgData.h > 0) {
      el.setAttribute('width', String(svgData.w * 0.8))
      el.setAttribute('height', String(svgData.h * 0.8))
    }
  }, [svgData])

  // 휠 이벤트는 passive: false 로 등록해야 preventDefault가 동작함
  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      setScale((s) => {
        const next = Math.min(Math.max(s * (e.deltaY < 0 ? 1.1 : 0.9), 0.1), 10)
        const r = next / s
        setOffset((o) => ({ x: mx - r * (mx - o.x), y: my - r * (my - o.y) }))
        return next
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  function onMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    if (e.button !== 0) return
    e.preventDefault()
    setIsDragging(true)
    dragStart.current = { mx: e.clientX, my: e.clientY, ox: offset.x, oy: offset.y }
  }

  function onMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    if (!dragStart.current) return
    setOffset({
      x: dragStart.current.ox + e.clientX - dragStart.current.mx,
      y: dragStart.current.oy + e.clientY - dragStart.current.my,
    })
  }

  function stopDrag() {
    setIsDragging(false)
    dragStart.current = null
  }

  const baseW = svgData.w > 0 ? svgData.w * 0.8 : 600
  const baseH = svgData.h > 0 ? svgData.h * 0.8 : 400

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-6"
      style={{ background: 'rgba(0,0,0,0.8)' }}
      onClick={onClose}
    >
      <div
        style={{
          position: 'relative',
          background: 'white',
          borderRadius: '1rem',
          boxShadow: '0 25px 50px rgba(0,0,0,0.25)',
          maxWidth: '90vw',
          maxHeight: '90vh',
          width: `${baseW + 64}px`,
          height: `${baseH + 52}px`,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* 헤더 */}
        <div style={{ padding: '10px 16px', textAlign: 'right', flexShrink: 0, borderBottom: '1px solid #f3f4f6' }}>
          <button
            className="w-8 h-8 bg-gray-100 hover:bg-gray-200 rounded-full flex items-center justify-center text-gray-600 font-bold transition-colors"
            style={{ display: 'inline-flex', border: 'none', cursor: 'pointer' }}
            onClick={onClose}
            aria-label="닫기"
          >
            ✕
          </button>
        </div>

        {/* Pan/Zoom 뷰포트 */}
        <div
          ref={viewportRef}
          style={{
            flex: 1,
            overflow: 'hidden',
            position: 'relative',
            cursor: isDragging ? 'grabbing' : 'grab',
            userSelect: 'none',
          }}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={stopDrag}
          onMouseLeave={stopDrag}
        >
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
              transformOrigin: '0 0',
            }}
          >
            <div ref={innerRef} />
          </div>
        </div>
      </div>
    </div>,
    document.body,
  )
}

export default function MermaidDiagram({ code }: { code: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [svgData, setSvgData] = useState<SvgData | null>(null)
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    if (!ref.current) return
    const id = `mermaid-${++counter}`
    const sanitized = sanitizeMermaid(code)
    mermaid.render(id, sanitized)
      .then(({ svg }) => {
        if (!ref.current) return
        ref.current.innerHTML = svg
        const svgEl = ref.current.querySelector('svg')
        if (!svgEl) return
        // 썸네일은 mermaid 기본 max-width CSS 그대로 유지 (작은 미리보기)
        const viewBox = svgEl.getAttribute('viewBox')
        let w = 0, h = 0
        if (viewBox) {
          const parts = viewBox.trim().split(/\s+/).map(Number)
          if (parts.length === 4 && parts[2] > 0 && parts[3] > 0) {
            w = parts[2]
            h = parts[3]
          }
        }
        setSvgData({ raw: svg, w, h })
      })
      .catch((e) => setError(String(e)))
  }, [code])

  if (error) {
    return (
      <div className="bg-red-50 border border-red-100 rounded-lg p-3 text-xs text-red-600">
        다이어그램 렌더링 실패: {error}
        <pre className="mt-2 whitespace-pre-wrap text-gray-500">{code}</pre>
      </div>
    )
  }

  return (
    <>
      <div
        ref={ref}
        className="overflow-x-auto my-2 cursor-zoom-in"
        onClick={() => svgData && setIsOpen(true)}
        title="클릭하면 크게 볼 수 있습니다"
      />
      {isOpen && svgData && (
        <MermaidModal svgData={svgData} onClose={() => setIsOpen(false)} />
      )}
    </>
  )
}
