import { useState } from "react";
// framer-motion removed — CSS animation used instead
import { X, ArrowLeft } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import OnboardingFlow from "./OnboardingFlow";
import { useTranslation } from "react-i18next";

/* ── Google "G" icon ── */
const GOOGLE_G = (
  <svg width="20" height="20" viewBox="0 0 48 48">
    <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
    <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
    <path fill="#FBBC05" d="M10.53 28.59a14.5 14.5 0 0 1 0-9.18l-7.98-6.19a24.01 24.01 0 0 0 0 21.56l7.98-6.19z"/>
    <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
  </svg>
);

/* ── View type ── */
type View = "login" | "register" | "email-login" | "email-register";

/* ── Shared styles ── */
const overlayStyle: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  zIndex: 1000,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backdropFilter: "blur(10px)",
  WebkitBackdropFilter: "blur(10px)",
  background: "rgba(0,0,0,0.45)",
};

const cardStyle: React.CSSProperties = {
  position: "relative",
  width: 460,
  maxWidth: "92vw",
  minHeight: 520,
  background: "var(--color-surface)",
  borderRadius: 20,
  boxShadow: "0 12px 48px rgba(0,0,0,0.18), 0 0 0 1px rgba(0,0,0,0.04)",
  padding: "44px 36px 36px",
  overflow: "hidden",
};

const headingStyle: React.CSSProperties = {
  fontFamily: "var(--font-display)",
  fontSize: 26,
  fontWeight: 600,
  color: "var(--color-ink)",
  textAlign: "center",
  margin: 0,
  lineHeight: 1.25,
};

const subtitleStyle: React.CSSProperties = {
  fontFamily: "var(--font-body)",
  fontSize: 15,
  color: "var(--color-ink-secondary)",
  textAlign: "center",
  margin: "8px 0 0",
};

const primaryBtnStyle: React.CSSProperties = {
  width: "100%",
  padding: "14px 0",
  background: "var(--color-leaf)",
  color: "white",
  border: "none",
  borderRadius: 12,
  fontSize: 15,
  fontWeight: 600,
  cursor: "pointer",
  fontFamily: "var(--font-body)",
  transition: "opacity 0.15s",
};

const googleBtnStyle: React.CSSProperties = {
  width: "100%",
  padding: "14px 0",
  background: "rgba(66, 133, 244, 0.08)",
  color: "#4285F4",
  border: "none",
  borderRadius: 12,
  fontSize: 15,
  fontWeight: 600,
  cursor: "pointer",
  fontFamily: "var(--font-body)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 10,
  transition: "background 0.15s",
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "12px 14px",
  background: "transparent",
  border: "1px solid var(--border-color)",
  borderRadius: 10,
  fontSize: 14,
  fontFamily: "var(--font-body)",
  color: "var(--color-ink)",
  outline: "none",
  transition: "border-color 0.15s",
  boxSizing: "border-box",
};

const linkBtnStyle: React.CSSProperties = {
  background: "none",
  border: "none",
  color: "var(--color-leaf)",
  fontWeight: 600,
  cursor: "pointer",
  fontFamily: "var(--font-body)",
  fontSize: 14,
  padding: 0,
};

const closeBtnStyle: React.CSSProperties = {
  position: "absolute",
  top: 16,
  right: 16,
  background: "none",
  border: "none",
  cursor: "pointer",
  color: "var(--color-ink-tertiary)",
  padding: 4,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  borderRadius: 8,
  transition: "color 0.15s",
};

const backBtnStyle: React.CSSProperties = {
  position: "absolute",
  top: 16,
  left: 16,
  background: "none",
  border: "none",
  cursor: "pointer",
  color: "var(--color-ink-tertiary)",
  padding: 4,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  borderRadius: 8,
  transition: "color 0.15s",
};

/* ── Divider ── */
function Divider() {
  const { t } = useTranslation();
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 14, margin: "20px 0" }}>
      <div style={{ flex: 1, height: 1, background: "var(--border-emphasis)" }} />
      <span style={{ fontSize: 12, fontFamily: "var(--font-body)", color: "var(--color-ink-tertiary)", fontWeight: 600, letterSpacing: 1, textTransform: "uppercase" }}>{t("auth.divider.or")}</span>
      <div style={{ flex: 1, height: 1, background: "var(--border-emphasis)" }} />
    </div>
  );
}

/* ── View wrapper — simple fade via CSS ── */
const viewStyle: React.CSSProperties = {
  animation: "authFadeIn 0.22s ease-out",
};

/* ── Main component ── */
export default function AuthGate() {
  const { t } = useTranslation();
  const { user, userProfile, loading, login, register, loginWithGoogle, showAuthGate, setShowAuthGate } = useAuth();

  const [view, setView] = useState<View>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (loading) return null;

  // Authenticated and onboarded → no gate
  if (user && userProfile?.onboarded) return null;

  // Authenticated but not onboarded → show onboarding
  if (user && userProfile && !userProfile.onboarded) {
    return (
      <div style={overlayStyle}>
        <div style={cardStyle}>
          <OnboardingFlow />
        </div>
      </div>
    );
  }

  // If user exists (profile still loading maybe), don't show auth gate
  if (user) return null;

  // Not authenticated: only show when triggered by "Log in" button
  if (!showAuthGate) return null;

  const resetForm = () => {
    setEmail("");
    setPassword("");
    setFirstName("");
    setLastName("");
    setError("");
    setSubmitting(false);
  };

  const switchView = (v: View) => {
    resetForm();
    setView(v);
  };

  const handleClose = () => {
    // Only allow closing if externally triggered (user exists scenario)
    // When no user, modal is mandatory — but we still wire the X for future use
    setShowAuthGate(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      if (view === "email-login") {
        await login(email, password);
      } else {
        if (!firstName.trim()) { setError(t("auth.errors.enterFirstName")); setSubmitting(false); return; }
        await register(email, password, firstName.trim(), lastName.trim());
      }
      setShowAuthGate(false);
    } catch (err: any) {
      const code = err?.code || "";
      if (code === "auth/email-already-in-use") setError(t("auth.errors.emailInUse"));
      else if (code === "auth/invalid-email") setError(t("auth.errors.invalidEmail"));
      else if (code === "auth/weak-password") setError(t("auth.errors.weakPassword"));
      else if (code === "auth/invalid-credential") setError(t("auth.errors.wrongCredentials"));
      else setError(err?.message || "Something went wrong");
    }
    setSubmitting(false);
  };

  const handleGoogleSignIn = async () => {
    setError("");
    try {
      await loginWithGoogle();
      setShowAuthGate(false);
    } catch (err: any) {
      if (err?.code !== "auth/popup-closed-by-user") {
        setError(err?.message || "Google Sign-In error");
      }
    }
  };

  /* ── Login method selection ── */
  const renderLogin = () => (
    <div key="login" style={viewStyle}>
      <div style={{ textAlign: "center", marginBottom: 28 }}>
        <img src="/Cavio Logo.png" alt="Cavio" style={{ width: 52, height: 52, marginBottom: 14, display: "block", marginLeft: "auto", marginRight: "auto" }} />
        <h2 style={headingStyle}>{t("auth.login.title")}</h2>
        <p style={subtitleStyle}>{t("auth.login.subtitle")}</p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <button
          style={primaryBtnStyle}
          onClick={() => switchView("email-login")}
          onMouseEnter={(e) => e.currentTarget.style.opacity = "0.88"}
          onMouseLeave={(e) => e.currentTarget.style.opacity = "1"}
        >
          {t("auth.login.emailBtn")}
        </button>
        <button
          style={googleBtnStyle}
          onClick={handleGoogleSignIn}
          onMouseEnter={(e) => e.currentTarget.style.background = "rgba(66, 133, 244, 0.14)"}
          onMouseLeave={(e) => e.currentTarget.style.background = "rgba(66, 133, 244, 0.08)"}
        >
          {GOOGLE_G}
          {t("auth.login.googleBtn")}
        </button>
      </div>

      <Divider />

      <div style={{ textAlign: "center", fontSize: 14, color: "var(--color-ink-secondary)", fontFamily: "var(--font-body)" }}>
        {t("auth.login.noAccount")}{" "}
        <button style={linkBtnStyle} onClick={() => switchView("register")}>
          {t("auth.login.createAccount")}
        </button>
      </div>
    </div>
  );

  /* ── Register method selection ── */
  const renderRegister = () => (
    <div key="register" style={viewStyle}>
      <div style={{ textAlign: "center", marginBottom: 28 }}>
        <img src="/Cavio Logo.png" alt="Cavio" style={{ width: 52, height: 52, marginBottom: 14, display: "block", marginLeft: "auto", marginRight: "auto" }} />
        <h2 style={headingStyle}>{t("auth.register.title")}</h2>
        <p style={subtitleStyle}>{t("auth.register.subtitle")}</p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <button
          style={primaryBtnStyle}
          onClick={() => switchView("email-register")}
          onMouseEnter={(e) => e.currentTarget.style.opacity = "0.88"}
          onMouseLeave={(e) => e.currentTarget.style.opacity = "1"}
        >
          {t("auth.register.emailBtn")}
        </button>
        <button
          style={googleBtnStyle}
          onClick={handleGoogleSignIn}
          onMouseEnter={(e) => e.currentTarget.style.background = "rgba(66, 133, 244, 0.14)"}
          onMouseLeave={(e) => e.currentTarget.style.background = "rgba(66, 133, 244, 0.08)"}
        >
          {GOOGLE_G}
          {t("auth.register.googleBtn")}
        </button>
      </div>

      <Divider />

      <div style={{ textAlign: "center", fontSize: 14, color: "var(--color-ink-secondary)", fontFamily: "var(--font-body)" }}>
        {t("auth.register.hasAccount")}{" "}
        <button style={linkBtnStyle} onClick={() => switchView("login")}>
          {t("auth.register.loginHere")}
        </button>
      </div>

      <p style={{ textAlign: "center", fontSize: 12, color: "var(--color-ink-tertiary)", fontFamily: "var(--font-body)", marginTop: 16, lineHeight: 1.5 }}>
        {t("auth.register.termsText")}{" "}
        <a href="/terms" style={{ color: "var(--color-leaf)", textDecoration: "underline" }}>{t("auth.register.termsLink")}</a>{" "}
        {t("auth.register.and")}{" "}
        <a href="/privacy" style={{ color: "var(--color-leaf)", textDecoration: "underline" }}>{t("auth.register.privacyLink")}</a>.
      </p>
    </div>
  );

  /* ── Email form (login or register) ── */
  const renderEmailForm = () => {
    const isLogin = view === "email-login";
    return (
      <div key="email-form" style={viewStyle}>
        <div style={{ textAlign: "center", marginBottom: 24 }}>
          <img src="/Cavio Logo.png" alt="Cavio" style={{ width: 52, height: 52, marginBottom: 14, display: "block", marginLeft: "auto", marginRight: "auto" }} />
          <h2 style={headingStyle}>{isLogin ? t("auth.emailForm.signIn") : t("auth.emailForm.createAccount")}</h2>
        </div>

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {!isLogin && (
            <div style={{ display: "flex", gap: 12 }}>
              <input
                style={inputStyle}
                placeholder={t("auth.emailForm.firstName")}
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                onFocus={(e) => e.currentTarget.style.borderColor = "var(--color-leaf)"}
                onBlur={(e) => e.currentTarget.style.borderColor = "var(--border-color)"}
              />
              <input
                style={inputStyle}
                placeholder={t("auth.emailForm.lastName")}
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
                onFocus={(e) => e.currentTarget.style.borderColor = "var(--color-leaf)"}
                onBlur={(e) => e.currentTarget.style.borderColor = "var(--border-color)"}
              />
            </div>
          )}
          <input
            style={inputStyle}
            type="email"
            placeholder={t("auth.emailForm.email")}
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            onFocus={(e) => e.currentTarget.style.borderColor = "var(--color-leaf)"}
            onBlur={(e) => e.currentTarget.style.borderColor = "var(--border-color)"}
            required
          />
          <input
            style={inputStyle}
            type="password"
            placeholder={t("auth.emailForm.password")}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onFocus={(e) => e.currentTarget.style.borderColor = "var(--color-leaf)"}
            onBlur={(e) => e.currentTarget.style.borderColor = "var(--border-color)"}
            required
            minLength={6}
          />

          {error && (
            <div style={{ color: "var(--color-high)", fontSize: 13, fontWeight: 500, fontFamily: "var(--font-body)" }}>
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting}
            style={{ ...primaryBtnStyle, opacity: submitting ? 0.7 : 1, marginTop: 4 }}
          >
            {submitting ? t("auth.emailForm.loading") : isLogin ? t("auth.emailForm.signIn") : t("auth.emailForm.createAccountBtn")}
          </button>
        </form>
      </div>
    );
  };

  const isEmailView = view === "email-login" || view === "email-register";

  return (
    <div style={overlayStyle}>
      <div style={cardStyle}>
        {/* Close button */}
        <button
          style={closeBtnStyle}
          onClick={handleClose}
          onMouseEnter={(e) => e.currentTarget.style.color = "var(--color-ink)"}
          onMouseLeave={(e) => e.currentTarget.style.color = "var(--color-ink-tertiary)"}
          aria-label="Close"
        >
          <X size={20} />
        </button>

        {/* Back button (email views only) */}
        {isEmailView && (
          <button
            style={backBtnStyle}
            onClick={() => switchView(view === "email-login" ? "login" : "register")}
            onMouseEnter={(e) => e.currentTarget.style.color = "var(--color-ink)"}
            onMouseLeave={(e) => e.currentTarget.style.color = "var(--color-ink-tertiary)"}
            aria-label="Back"
          >
            <ArrowLeft size={20} />
          </button>
        )}

        {view === "login" && renderLogin()}
        {view === "register" && renderRegister()}
        {isEmailView && renderEmailForm()}
      </div>
    </div>
  );
}
