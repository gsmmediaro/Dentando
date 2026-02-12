import type { AnalysisResult } from "../api/client";
import FindingsTable from "./FindingsTable";

interface Props {
  result: AnalysisResult;
  imageInline?: boolean;
}

export default function VerdictCard({ result, imageInline = true }: Props) {
  return (
    <div style={{ width: "100%", marginTop: 32 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
        {imageInline && (
          <span className={`verdict-badge ${result.suspicion_level.toLowerCase()}`}>
            {result.suspicion_level}
          </span>
        )}
        <span style={{ fontSize: 13, color: "var(--color-ink-tertiary)" }}>
          {result.model_name} &middot; {result.modality} &middot; {result.num_detections} {result.num_detections === 1 ? "detectie" : "detectii"} &middot; {result.turnaround_s}s
        </span>
      </div>
      {imageInline && (
        <div style={{
          width: "100%",
          background: "#1a1a1a",
          borderRadius: 10,
          padding: 16,
          marginBottom: 20,
        }}>
          <img
            style={{ width: "100%", display: "block", borderRadius: 6 }}
            src={result.annotated_image_url}
            alt="Radiografie adnotata"
          />
        </div>
      )}
      <FindingsTable detections={result.detections} />
    </div>
  );
}
