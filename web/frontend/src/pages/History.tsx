import { useEffect, useState } from "react";
import { getHistoryFromFirestore, type ScanRecord } from "../api/client";
import { useAuth } from "../contexts/AuthContext";

function badgeLevel(s: string): string {
  return `verdict-badge ${s.toLowerCase()}`;
}

const PAGE_SIZE = 20;

export default function History() {
  const { user } = useAuth();
  const [records, setRecords] = useState<ScanRecord[]>([]);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);

  useEffect(() => {
    if (!user) return;
    // Fetch enough records for pagination (simple client-side approach)
    getHistoryFromFirestore(user.uid, (page + 1) * PAGE_SIZE + 1).then((all) => {
      const start = page * PAGE_SIZE;
      const slice = all.slice(start, start + PAGE_SIZE);
      setRecords(slice);
      setHasMore(all.length > start + PAGE_SIZE);
    });
  }, [user, page]);

  const thStyle: React.CSSProperties = {
    textAlign: "left",
    padding: "12px 16px",
    fontWeight: 500,
    color: "var(--color-ink-tertiary)",
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    borderBottom: "1px solid var(--border-color)",
  };

  const tdStyle: React.CSSProperties = {
    padding: "12px 16px",
    borderBottom: "1px solid var(--border-color)",
  };

  return (
    <div style={{
      flex: 1,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: window.innerWidth <= 768 ? "24px 16px" : "40px 32px",
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
        Scan history
      </h1>

      <div className="mobile-table-scroll" style={{
        width: "100%",
        background: "var(--color-surface)",
        border: "1px solid var(--border-color)",
        borderRadius: 10,
        overflow: "hidden",
      }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr>
              {["Date", "Patient", "File", "Suspicion", "Confidence", "Detections", "Modality", "Speed"].map(h => (
                <th key={h} style={thStyle}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {records.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ textAlign: "center", color: "var(--color-ink-tertiary)", padding: 48 }}>
                  No scans.
                </td>
              </tr>
            ) : (
              records.map((r) => (
                <tr key={r.id}>
                  <td style={{ ...tdStyle, color: "var(--color-ink-tertiary)" }}>{new Date(r.timestamp).toLocaleString()}</td>
                  <td style={{ ...tdStyle, fontWeight: 500 }}>{r.patient_name || "—"}</td>
                  <td style={{ ...tdStyle, color: "var(--color-ink-secondary)" }}>{r.filename}</td>
                  <td style={tdStyle}><span className={badgeLevel(r.suspicion)}>{r.suspicion}</span></td>
                  <td style={tdStyle}>{(r.confidence * 100).toFixed(1)}%</td>
                  <td style={tdStyle}>{r.detections_count}</td>
                  <td style={{ ...tdStyle, color: "var(--color-ink-tertiary)" }}>{r.modality}</td>
                  <td style={{ ...tdStyle, color: "var(--color-ink-tertiary)" }}>{r.turnaround_s}s</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginTop: 24 }}>
        {[
          { label: "Back", disabled: page === 0, onClick: () => setPage(Math.max(0, page - 1)) },
          { label: "Next", disabled: !hasMore, onClick: () => setPage(page + 1) },
        ].map(btn => (
          <button
            key={btn.label}
            disabled={btn.disabled}
            onClick={btn.onClick}
            style={{
              display: "inline-flex",
              alignItems: "center",
              padding: "10px 20px",
              borderRadius: 6,
              fontSize: 14,
              fontWeight: 500,
              background: "transparent",
              color: "var(--color-ink-secondary)",
              border: "1px solid var(--border-emphasis)",
              cursor: btn.disabled ? "not-allowed" : "pointer",
              opacity: btn.disabled ? 0.4 : 1,
              transition: "background 0.15s",
              fontFamily: "var(--font-body)",
            }}
          >
            {btn.label}
          </button>
        ))}
      </div>
    </div>
  );
}
