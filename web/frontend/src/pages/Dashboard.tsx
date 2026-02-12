import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getStatsFromFirestore, getHistoryFromFirestore, type DailyStats, type ScanRecord } from "../api/client";
import { useAuth } from "../contexts/AuthContext";
import StatCard from "../components/StatCard";

function badgeLevel(s: string): string {
  return `verdict-badge ${s.toLowerCase()}`;
}

export default function Dashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState<DailyStats>({ total: 0, high: 0, review: 0, avg_turnaround: 0 });
  const [recent, setRecent] = useState<ScanRecord[]>([]);

  useEffect(() => {
    if (!user) return;
    getStatsFromFirestore(user.uid).then(setStats);
    getHistoryFromFirestore(user.uid, 10).then(setRecent);
  }, [user]);

  return (
    <div style={{
      flex: 1,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "40px 32px",
      maxWidth: 1000,
      width: "100%",
      margin: "0 auto",
    }}>
      <h1 style={{
        fontFamily: "var(--font-display)",
        fontSize: 28,
        fontWeight: 400,
        color: "var(--color-ink)",
        marginBottom: 32,
        textAlign: "center",
      }}>
        Welcome
      </h1>

      <div style={{
        display: "flex",
        gap: 32,
        padding: "16px 0",
        marginBottom: 24,
        borderBottom: "1px solid var(--border-color)",
        width: "100%",
      }}>
        <StatCard label="Scans today" value={stats.total} />
        <StatCard label="High suspicion" value={stats.high} color="var(--color-high)" />
        <StatCard label="Needs review" value={stats.review} color="var(--color-review)" />
        <StatCard label="Avg turnaround" value={`${stats.avg_turnaround}s`} />
      </div>

      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        width: "100%",
        marginBottom: 16,
      }}>
        <span style={{ fontWeight: 500 }}>Recent scans</span>
        <Link
          to="/analyze"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            padding: "8px 16px",
            background: "var(--color-leaf)",
            color: "#faf9f6",
            fontSize: 13,
            fontWeight: 500,
            borderRadius: 6,
            textDecoration: "none",
            transition: "background 0.15s",
          }}
        >
          + New scan
        </Link>
      </div>

      <div style={{
        width: "100%",
        background: "var(--color-surface)",
        border: "1px solid var(--border-color)",
        borderRadius: 10,
        overflow: "hidden",
      }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr>
              {["Time", "Patient", "File", "Suspicion", "Confidence", "Detections"].map(h => (
                <th key={h} style={{
                  textAlign: "left",
                  padding: "12px 16px",
                  fontWeight: 500,
                  color: "var(--color-ink-tertiary)",
                  fontSize: 11,
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  borderBottom: "1px solid var(--border-color)",
                }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {recent.length === 0 ? (
              <tr>
                <td colSpan={6} style={{ textAlign: "center", color: "var(--color-ink-tertiary)", padding: 48 }}>
                  No scans yet. Start by analyzing an X-ray.
                </td>
              </tr>
            ) : (
              recent.map((r) => (
                <tr key={r.id}>
                  <td style={{ padding: "12px 16px", color: "var(--color-ink-tertiary)", borderBottom: "1px solid var(--border-color)" }}>{new Date(r.timestamp).toLocaleTimeString()}</td>
                  <td style={{ padding: "12px 16px", fontWeight: 500, borderBottom: "1px solid var(--border-color)" }}>{r.patient_name || "—"}</td>
                  <td style={{ padding: "12px 16px", color: "var(--color-ink-secondary)", borderBottom: "1px solid var(--border-color)" }}>{r.filename}</td>
                  <td style={{ padding: "12px 16px", borderBottom: "1px solid var(--border-color)" }}><span className={badgeLevel(r.suspicion)}>{r.suspicion}</span></td>
                  <td style={{ padding: "12px 16px", borderBottom: "1px solid var(--border-color)" }}>{(r.confidence * 100).toFixed(1)}%</td>
                  <td style={{ padding: "12px 16px", borderBottom: "1px solid var(--border-color)" }}>{r.detections_count}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
