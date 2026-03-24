import { useTranslation } from "react-i18next";

export default function Privacy() {
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
        {t("privacy.title")}
      </h1>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("privacy.s1h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("privacy.s1p")}</p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("privacy.s2h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("privacy.s2p")}</p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("privacy.s3h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("privacy.s3p")}</p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("privacy.s4h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("privacy.s4p")}</p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("privacy.s5h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("privacy.s5p")}</p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("privacy.s6h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("privacy.s6p")}</p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>{t("privacy.s7h")}</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>{t("privacy.s7p")}</p>
    </div>
  );
}
