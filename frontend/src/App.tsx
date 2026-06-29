import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useStore } from './store/useStore'
import Login from './pages/Login'
import Register from './pages/Register'
import CompanySetup from './pages/CompanySetup'
import Chat from './pages/Chat'

function RequireAuth({ children }: { children: React.ReactNode }) {
  const token = useStore((s) => s.token)
  if (!token) return <Navigate to="/login" replace />
  return <>{children}</>
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route
          path="/company-setup"
          element={
            <RequireAuth>
              <CompanySetup />
            </RequireAuth>
          }
        />
        {/* / 는 비로그인도 접근 가능. needs_company_info 에이전트 클릭 시에만 모달 처리 */}
        <Route path="/" element={<Chat />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
