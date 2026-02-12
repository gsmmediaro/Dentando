export default function Terms() {
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
        Terms and Conditions
      </h1>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>1. Acceptance of terms</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        By using the Quinn platform, you accept these terms and conditions in full. If you do not agree with any provision, please do not use the platform.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>2. Service description</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        Quinn is an AI-powered support tool designed to assist dental professionals in analyzing dental radiographs. The platform does not replace professional clinical diagnosis.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>3. Proper use</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        Results provided by the platform are indicative and must be validated by a qualified professional. The user is responsible for all clinical decisions based on the information provided.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>4. User account</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        You are responsible for keeping your account and password confidential. Any activity performed through your account is your responsibility.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>5. Trial period</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        New users receive a 7-day free trial. After the trial period, access to premium features may be restricted.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>6. Limitation of liability</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        Quinn is not responsible for clinical decisions made based on platform results. The service is provided "as is," without warranties of any kind.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>7. Changes to terms</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        We reserve the right to modify these terms at any time. Users will be notified through the platform about significant changes.
      </p>
    </div>
  );
}
