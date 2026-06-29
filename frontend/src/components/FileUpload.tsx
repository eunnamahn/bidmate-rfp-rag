import { useRef, useState } from 'react'
import { uploadFile, fetchDocuments } from '../api/documents'
import { useStore } from '../store/useStore'

export default function FileUpload() {
  const inputRef = useRef<HTMLInputElement>(null)
  const setDocuments = useStore((s) => s.setDocuments)
  const setActiveFilename = useStore((s) => s.setActiveFilename)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')

  async function handleFile(file: File) {
    setError('')
    setUploading(true)
    try {
      const data = await uploadFile(file)
      const docs = await fetchDocuments()
      setDocuments(docs)
      setActiveFilename(data.파일명)
    } catch {
      setError('업로드에 실패했습니다.')
    } finally {
      setUploading(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  return (
    <div>
      <div
        className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
          uploading ? 'border-primary-300 bg-primary-50' : 'border-gray-200 hover:border-primary-400 hover:bg-gray-50'
        }`}
        onClick={() => !uploading && inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        {uploading ? (
          <p className="text-xs text-primary-600 font-medium">파싱·인덱싱 중...</p>
        ) : (
          <>
            <svg className="w-6 h-6 text-gray-300 mx-auto mb-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <p className="text-xs text-gray-400">HWP · HWPX · PDF</p>
            <p className="text-xs text-gray-300">드래그 또는 클릭</p>
          </>
        )}
      </div>
      <input
        ref={inputRef} type="file"
        accept=".hwp,.hwpx,.pdf"
        className="hidden"
        onChange={handleChange}
      />
      {error && <p className="text-xs text-red-500 mt-1">{error}</p>}
    </div>
  )
}
