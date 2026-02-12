export default function StatCard({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <div style={{
        fontSize: 22,
        fontWeight: 600,
        fontFamily: "var(--font-display)",
        color: color || "var(--color-ink)",
      }}>
        {value}
      </div>
      <div style={{
        fontSize: 11,
        fontWeight: 500,
        color: "var(--color-ink-tertiary)",
        textTransform: "uppercase",
        letterSpacing: "0.06em",
      }}>
        {label}
      </div>
    </div>
  );
}
