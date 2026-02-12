import type { Detection } from "../api/client";

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
  color: "var(--color-ink-secondary)",
};

export default function FindingsTable({ detections }: { detections: Detection[] }) {
  if (detections.length === 0) {
    return <p style={{ color: "var(--color-ink-tertiary)", fontSize: 13 }}>Nicio detectie gasita.</p>;
  }

  return (
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
            <th style={thStyle}>Clasa</th>
            <th style={thStyle}>Incredere</th>
            <th style={thStyle}>Locatie</th>
          </tr>
        </thead>
        <tbody>
          {detections.map((d, i) => (
            <tr key={i}>
              <td style={{ ...tdStyle, fontWeight: 500, color: "var(--color-ink)" }}>{d.class_name}</td>
              <td style={tdStyle}>{(d.confidence * 100).toFixed(1)}%</td>
              <td style={{ ...tdStyle, fontFamily: "monospace", fontSize: 11, color: "var(--color-ink-tertiary)" }}>
                [{d.bbox.map((v) => v.toFixed(0)).join(", ")}]
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
