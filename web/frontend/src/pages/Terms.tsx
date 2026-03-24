import { useTranslation } from "react-i18next";

export default function Terms() {
  const { t } = useTranslation();
  return (
    <div style={{
      maxWidth: 700,
      margin: "0 auto",
      padding: "48px 32px",
      fontFamily: "var(--font-body)",
      color: "var(--color-ink)",
      lineHeight: 1.7,
    }}>
      <h1 style={{
        fontFamily: "var(--font-display)",
        fontSize: 28,
        fontWeight: 400,
        marginBottom: 32,
      }}>
        {t("terms.title")}
      </h1>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("terms.s1h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("terms.s1p")}</p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("terms.s2h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("terms.s2p")}</p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("terms.s3h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("terms.s3p")}</p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("terms.s4h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("terms.s4p")}</p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("terms.s5h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("terms.s5p")}</p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("terms.s6h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("terms.s6p")}</p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("terms.s7h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("terms.s7p")}</p>
    </div>
  );
}
