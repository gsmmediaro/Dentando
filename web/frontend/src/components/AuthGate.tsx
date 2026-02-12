import { useState } from "react";
import { useAuth } from "../contexts/AuthContext";
import OnboardingFlow from "./OnboardingFlow";

const GOOGLE_G = (
  <svg width="18" height="18" viewBox="0 0 48 48">
    <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
    <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
    <path fill="#FBBC05" d="M10.53 28.59a14.5 14.5 0 0 1 0-9.18l-7.98-6.19a24.01 24.01 0 0 0 0 21.56l7.98-6.19z"/>
    <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
  </svg>
);

const overlayStyle: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  zIndex: 1000,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backdropFilter: "blur(8px)",
  WebkitBackdropFilter: "blur(8px)",
  background: "rgba(0,0,0,0.4)",
};

const cardStyle: React.CSSProperties = {
  width: 440,
  maxWidth: "90vw",
  background: "var(--color-bg)",
  borderRadius: 16,
  border: "1px solid var(--border-color)",
  boxShadow: "0 8px 40px rgba(0,0,0,0.15)",
  padding: "36px 32px 28px",
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "10px 14px",
  background: "transparent",
  border: "1px solid var(--border-color)",
  borderRadius: 8,
  fontSize: 14,
  fontFamily: "var(--font-body)",
  color: "var(--color-ink)",
  outline: "none",
  transition: "border-color 0.15s",
};

const primaryBtnStyle: React.CSSProperties = {
  width: "100%",
  padding: "11px 0",
  background: "var(--color-leaf)",
  color: "white",
  border: "none",
  borderRadius: 8,
  fontSize: 14,
  fontWeight: 600,
  cursor: "pointer",
  fontFamily: "var(--font-body)",
  transition: "background 0.15s",
};

const googleBtnStyle: React.CSSProperties = {
  width: "100%",
  padding: "10px 0",
  background: "white",
  color: "var(--color-ink)",
  border: "1px solid var(--border-emphasis)",
  borderRadius: 8,
  fontSize: 14,
  fontWeight: 500,
  cursor: "pointer",
  fontFamily: "var(--font-body)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 10,
  transition: "background 0.15s",
};

export default function AuthGate() {
  const { user, userProfile, loading, login, register, loginWithGoogle } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      if (mode === "login") {
        await login(email, password);
      } else {
        if (!firstName.trim()) { setError("Introdu prenumele"); setSubmitting(false); return; }
        await register(email, password, firstName.trim(), lastName.trim());
      }
    } catch (err: any) {
      const code = err?.code || "";
      if (code === "auth/email-already-in-use") setError("Acest email este deja folosit");
      else if (code === "auth/invalid-email") setError("Email invalid");
      else if (code === "auth/weak-password") setError("Parola trebuie sa aiba minim 6 caractere");
      else if (code === "auth/invalid-credential") setError("Email sau parola gresita");
      else setError(err?.message || "Eroare necunoscuta");
    }
    setSubmitting(false);
  };

  const handleGoogleSignIn = async () => {
    setError("");
    try {
      await loginWithGoogle();
    } catch (err: any) {
      if (err?.code !== "auth/popup-closed-by-user") {
        setError(err?.message || "Eroare Google Sign-In");
      }
    }
  };

  return (
    <div style={overlayStyle}>
      <div style={cardStyle}>
        <h2 style={{
          fontFamily: "var(--font-display)",
          fontSize: 24,
          fontWeight: 400,
          color: "var(--color-ink)",
          textAlign: "center",
          marginBottom: 24,
        }}>
          {mode === "login" ? "Bine ai revenit" : "Creeaza cont"}
        </h2>

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {mode === "register" && (
            <div style={{ display: "flex", gap: 12 }}>
              <input
                style={inputStyle}
                placeholder="Prenume"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                onFocus={(e) => e.currentTarget.style.borderColor = "var(--color-leaf)"}
                onBlur={(e) => e.currentTarget.style.borderColor = "var(--border-color)"}
              />
              <input
                style={inputStyle}
                placeholder="Nume"
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
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            onFocus={(e) => e.currentTarget.style.borderColor = "var(--color-leaf)"}
            onBlur={(e) => e.currentTarget.style.borderColor = "var(--border-color)"}
            required
          />
          <input
            style={inputStyle}
            type="password"
            placeholder="Parola"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onFocus={(e) => e.currentTarget.style.borderColor = "var(--color-leaf)"}
            onBlur={(e) => e.currentTarget.style.borderColor = "var(--border-color)"}
            required
            minLength={6}
          />

          {error && (
            <div style={{ color: "var(--color-high)", fontSize: 13, fontWeight: 500 }}>
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting}
            style={{ ...primaryBtnStyle, opacity: submitting ? 0.7 : 1 }}
          >
            {submitting ? "Se incarca..." : mode === "login" ? "Intra in cont" : "Creeaza cont"}
          </button>
        </form>

        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          margin: "16px 0",
        }}>
          <div style={{ flex: 1, height: 1, background: "var(--border-emphasis)" }} />
          <span style={{ fontSize: 12, color: "var(--color-ink-tertiary)" }}>sau</span>
          <div style={{ flex: 1, height: 1, background: "var(--border-emphasis)" }} />
        </div>

        <button
          onClick={handleGoogleSignIn}
          style={googleBtnStyle}
          onMouseEnter={(e) => e.currentTarget.style.background = "var(--color-surface-hover)"}
          onMouseLeave={(e) => e.currentTarget.style.background = "white"}
        >
          {GOOGLE_G}
          Continua cu Google
        </button>

        <div style={{ textAlign: "center", marginTop: 20, fontSize: 13, color: "var(--color-ink-secondary)" }}>
          {mode === "login" ? (
            <>
              Nu ai cont?{" "}
              <button
                onClick={() => { setMode("register"); setError(""); }}
                style={{
                  background: "none",
                  border: "none",
                  color: "var(--color-leaf)",
                  fontWeight: 600,
                  cursor: "pointer",
                  fontFamily: "var(--font-body)",
                  fontSize: 13,
                }}
              >
                Inregistreaza-te
              </button>
            </>
          ) : (
            <>
              Ai deja cont?{" "}
              <button
                onClick={() => { setMode("login"); setError(""); }}
                style={{
                  background: "none",
                  border: "none",
                  color: "var(--color-leaf)",
                  fontWeight: 600,
                  cursor: "pointer",
                  fontFamily: "var(--font-body)",
                  fontSize: 13,
                }}
              >
                Intra in cont
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
