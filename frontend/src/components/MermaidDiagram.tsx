import { useEffect, useRef, useState } from 'react'
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

export default function MermaidDiagram({ code }: { code: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!ref.current) return
    const id = `mermaid-${++counter}`
    const sanitized = sanitizeMermaid(code)
    mermaid.render(id, sanitized)
      .then(({ svg }) => { if (ref.current) ref.current.innerHTML = svg })
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

  return <div ref={ref} className="overflow-x-auto my-2" />
}
