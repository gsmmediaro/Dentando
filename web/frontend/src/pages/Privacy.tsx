export default function Privacy() {
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
        Privacy Policy
      </h1>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>1. Data collected</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        We collect the following data: name, email address, organization information, and uploaded radiograph data for analysis. This data is required to provide the service.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>2. Data use</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        Your data is used exclusively to provide and improve Quinn services. We do not sell or share personal data with third parties without your explicit consent.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>3. Data storage</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        Data is securely stored on Firebase (Google Cloud) servers. Each user can access only their own data. We implement appropriate security measures to protect information.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>4. Patient data</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        Radiographs and patient data are stored securely and are accessible only to the uploading user. We do not use patient data to train AI models without explicit consent.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>5. User rights</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        You have the right to access, modify, or delete your personal data at any time. For personal data requests, contact us at our support email address.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>6. Cookies</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        The platform uses essential cookies for authentication and proper functionality. We do not use tracking or advertising cookies.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>7. Contact</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        For any questions regarding this privacy policy or your data, feel free to contact us.
      </p>
    </div>
  );
}
