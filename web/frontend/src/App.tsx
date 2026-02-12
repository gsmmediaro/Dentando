import { useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import AuthGate from "./components/AuthGate";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import AnalyzeScan from "./pages/AnalyzeScan";
import History from "./pages/History";
import Terms from "./pages/Terms";
import Privacy from "./pages/Privacy";
import type { ScanRecord } from "./api/client";

export default function App() {
  const [patientScans, setPatientScans] = useState<ScanRecord[] | null>(null);
  const [patientName, setPatientName] = useState("");

  const handleSelectPatient = (scans: ScanRecord[], name: string) => {
    setPatientScans(scans);
    setPatientName(name);
  };

  return (
    <AuthProvider>
      <Layout onSelectPatient={handleSelectPatient}>
        <Routes>
          <Route path="/" element={<Navigate to="/analyze" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analyze" element={<AnalyzeScan />} />
          <Route path="/history" element={<History />} />
          <Route path="/terms" element={<Terms />} />
          <Route path="/privacy" element={<Privacy />} />
          <Route path="*" element={<Navigate to="/analyze" replace />} />
        </Routes>
      </Layout>
      <AuthGate />
    </AuthProvider>
  );
}
