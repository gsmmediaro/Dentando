import { Routes, Route, Navigate } from "react-router-dom";
import { Toaster } from "sonner";
import { AuthProvider } from "./contexts/AuthContext";
import AuthGate from "./components/AuthGate";
import Layout from "./components/Layout";
import AnalyzeScan from "./pages/AnalyzeScan";
import History from "./pages/History";
import Settings from "./pages/Settings";
import Terms from "./pages/Terms";
import Privacy from "./pages/Privacy";

export default function App() {
  return (
    <AuthProvider>
      <Toaster
        position="top-center"
        toastOptions={{
          style: {
            background: "var(--color-ink)",
            color: "white",
            border: "none",
            borderRadius: 12,
            fontSize: 14,
            fontFamily: "var(--font-body)",
            padding: "14px 20px",
            boxShadow: "0 8px 32px rgba(0,0,0,0.18)",
          },
        }}
      />
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/analyze" replace />} />
          <Route path="/analyze" element={<AnalyzeScan />} />
          <Route path="/history" element={<History />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/terms" element={<Terms />} />
          <Route path="/privacy" element={<Privacy />} />
          <Route path="*" element={<Navigate to="/analyze" replace />} />
        </Routes>
      </Layout>
      <AuthGate />
    </AuthProvider>
  );
}
