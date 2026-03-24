import { useAuth } from "../contexts/AuthContext";

export default function Settings() {
  const { user, userProfile } = useAuth();
  const isMobile = window.innerWidth <= 768;

  const displayName = userProfile
    ? `${userProfile.firstName} ${userProfile.lastName}`.trim()
    : user?.displayName || "";
  const email = user?.email || "";

  return (
    <div style={{
      flex: 1,
      display: "flex",
      flexDirection: "column",
      padding: isMobile ? "32px 20px" : "60px 32px",
      maxWidth: 680,
      width: "100%",
      margin: "0 auto",
    }}>
      <h1 style={{
        fontFamily: "var(--font-display)",
        fontSize: isMobile ? 32 : 42,
        fontWeight: 400,
        color: "var(--color-ink)",
        marginBottom: 32,
        textWrap: "balance",
      }}>
        Account Settings
      </h1>

      {/* Account Information card */}
      <div style={{
        background: "var(--color-surface)",
        borderRadius: 16,
        boxShadow: "0 0 0 1px rgba(45, 42, 36, 0.06), 0 1px 2px rgba(0,0,0,0.03)",
        padding: isMobile ? "24px 20px" : "28px 32px",
      }}>
        <div style={{ marginBottom: 24 }}>
          <h2 style={{
            fontFamily: "var(--font-display)",
            fontSize: 18,
            fontWeight: 500,
            color: "var(--color-ink)",
            margin: 0,
            lineHeight: 1.3,
          }}>
            Account Information
          </h2>
          <p style={{
            fontSize: 13,
            color: "var(--color-ink-tertiary)",
            margin: "4px 0 0",
            fontFamily: "var(--font-body)",
          }}>
            Manage & update your profile information
          </p>
        </div>

        <div style={{ borderTop: "1px solid var(--border-color)" }}>
          {/* Name row */}
          <div style={{
            display: "flex",
            alignItems: isMobile ? "flex-start" : "center",
            flexDirection: isMobile ? "column" : "row",
            justifyContent: "space-between",
            padding: "20px 0",
            borderBottom: "1px solid var(--border-color)",
            gap: isMobile ? 4 : 0,
          }}>
            <span style={{
              fontSize: 14,
              fontWeight: 600,
              color: "var(--color-ink)",
              fontFamily: "var(--font-body)",
            }}>
              Name
            </span>
            <span style={{
              fontSize: 14,
              color: "var(--color-ink-secondary)",
              fontFamily: "var(--font-body)",
            }}>
              {displayName || "—"}
            </span>
          </div>

          {/* Email row */}
          <div style={{
            display: "flex",
            alignItems: isMobile ? "flex-start" : "center",
            flexDirection: isMobile ? "column" : "row",
            justifyContent: "space-between",
            padding: "20px 0",
            gap: isMobile ? 4 : 0,
          }}>
            <span style={{
              fontSize: 14,
              fontWeight: 600,
              color: "var(--color-ink)",
              fontFamily: "var(--font-body)",
            }}>
              Email
            </span>
            <span style={{
              fontSize: 14,
              color: "var(--color-ink-secondary)",
              fontFamily: "var(--font-body)",
            }}>
              {email || "—"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
