import { useState } from "react";
import { doc, updateDoc, serverTimestamp } from "firebase/firestore";
import * as SelectPrimitive from "@radix-ui/react-select";
import { ChevronDown, X } from "lucide-react";
import { db } from "../firebase";
import { useAuth } from "../contexts/AuthContext";
import quinnLogo from "../../../../quinnnlogo.svg";

const SPECIALITIES = [
  "General dentist",
  "Orthodontist",
  "Endodontist",
  "Oral surgeon",
  "Pediatric dentist",
  "Periodontist",
  "Prosthodontist",
  "Other",
];

const ROLES = ["Founder", "Dentist", "Assistant", "Manager", "Other"];
const ORG_SIZES = ["Solo", "2-5", "6-10", "10+"];

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

const triggerStyle = (hasValue: boolean): React.CSSProperties => ({
  width: "100%",
  padding: "10px 14px",
  paddingRight: 36,
  background: "transparent",
  border: "1px solid var(--border-color)",
  borderRadius: 8,
  fontSize: 14,
  fontFamily: "var(--font-body)",
  color: hasValue ? "var(--color-ink)" : "var(--color-ink-tertiary)",
  cursor: "pointer",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  textAlign: "left",
  transition: "border-color 0.15s",
});

const contentStyle: React.CSSProperties = {
  background: "var(--color-surface)",
  border: "1px solid var(--border-emphasis)",
  borderRadius: 10,
  boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
  overflow: "hidden",
  zIndex: 1100,
  minWidth: "var(--radix-select-trigger-width)",
};

const itemStyle: React.CSSProperties = {
  padding: "8px 14px",
  fontSize: 13,
  color: "var(--color-ink)",
  borderRadius: 6,
  cursor: "pointer",
  outline: "none",
  transition: "background 0.1s",
  fontFamily: "var(--font-body)",
};

function StyledSelect({
  value,
  onValueChange,
  placeholder,
  options,
}: {
  value: string;
  onValueChange: (v: string) => void;
  placeholder: string;
  options: string[];
}) {
  return (
    <SelectPrimitive.Root value={value || undefined} onValueChange={onValueChange}>
      <SelectPrimitive.Trigger style={triggerStyle(!!value)}>
        <SelectPrimitive.Value placeholder={placeholder} />
        <SelectPrimitive.Icon>
          <ChevronDown size={14} style={{ color: "var(--color-ink-tertiary)" }} />
        </SelectPrimitive.Icon>
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Content
          style={contentStyle}
          position="popper"
          sideOffset={6}
          side="bottom"
          align="start"
        >
          <SelectPrimitive.Viewport style={{ padding: 4 }}>
            {options.map((opt) => (
              <SelectPrimitive.Item
                key={opt}
                value={opt}
                style={itemStyle}
                className="data-[highlighted]:bg-leaf-subtle"
              >
                <SelectPrimitive.ItemText>{opt}</SelectPrimitive.ItemText>
              </SelectPrimitive.Item>
            ))}
          </SelectPrimitive.Viewport>
        </SelectPrimitive.Content>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  );
}

function TermsContent() {
  const h2: React.CSSProperties = {
    fontSize: 15,
    fontWeight: 600,
    marginTop: 20,
    marginBottom: 6,
    fontFamily: "var(--font-display)",
  };
  const p: React.CSSProperties = { fontSize: 13, color: "var(--color-ink-secondary)", marginBottom: 12, lineHeight: 1.65 };
  return (
    <>
      <h2 style={h2}>1. Acceptance of terms</h2>
      <p style={p}>By using the Quinn platform, you accept these terms and conditions in full. If you do not agree with any provision, please do not use the platform.</p>
      <h2 style={h2}>2. Service description</h2>
      <p style={p}>Quinn is an AI-powered support tool designed to assist dental professionals in analyzing dental radiographs. The platform does not replace professional clinical diagnosis.</p>
      <h2 style={h2}>3. Proper use</h2>
      <p style={p}>Results provided by the platform are indicative and must be validated by a qualified professional. The user is responsible for all clinical decisions based on the information provided.</p>
      <h2 style={h2}>4. User account</h2>
      <p style={p}>You are responsible for keeping your account and password confidential. Any activity performed through your account is your responsibility.</p>
      <h2 style={h2}>5. Trial period</h2>
      <p style={p}>New users receive a 7-day free trial. After the trial period, access to premium features may be restricted.</p>
      <h2 style={h2}>6. Limitation of liability</h2>
      <p style={p}>Quinn is not responsible for clinical decisions made based on platform results. The service is provided "as is," without warranties of any kind.</p>
      <h2 style={h2}>7. Changes to terms</h2>
      <p style={p}>We reserve the right to modify these terms at any time. Users will be notified through the platform about significant changes.</p>
    </>
  );
}

function PrivacyContent() {
  const h2: React.CSSProperties = {
    fontSize: 15,
    fontWeight: 600,
    marginTop: 20,
    marginBottom: 6,
    fontFamily: "var(--font-display)",
  };
  const p: React.CSSProperties = { fontSize: 13, color: "var(--color-ink-secondary)", marginBottom: 12, lineHeight: 1.65 };
  return (
    <>
      <h2 style={h2}>1. Data collected</h2>
      <p style={p}>We collect the following data: name, email address, organization information, and uploaded radiograph data for analysis. This data is required to provide the service.</p>
      <h2 style={h2}>2. Data use</h2>
      <p style={p}>Your data is used exclusively to provide and improve Quinn services. We do not sell or share personal data with third parties without your explicit consent.</p>
      <h2 style={h2}>3. Data storage</h2>
      <p style={p}>Data is securely stored on Firebase (Google Cloud) servers. Each user can access only their own data. We implement appropriate security measures to protect information.</p>
      <h2 style={h2}>4. Patient data</h2>
      <p style={p}>Radiographs and patient data are stored securely and are accessible only to the uploading user. We do not use patient data to train AI models without explicit consent.</p>
      <h2 style={h2}>5. User rights</h2>
      <p style={p}>You have the right to access, modify, or delete your personal data at any time. For personal data requests, contact us at our support email address.</p>
      <h2 style={h2}>6. Cookies</h2>
      <p style={p}>The platform uses essential cookies for authentication and proper functionality. We do not use tracking or advertising cookies.</p>
    </>
  );
}

export default function OnboardingFlow() {
  const { user, refreshProfile } = useAuth();
  const [step, setStep] = useState(0);
  const [speciality, setSpeciality] = useState("");
  const [role, setRole] = useState("");
  const [agreed, setAgreed] = useState(false);
  const [orgName, setOrgName] = useState("");
  const [orgSize, setOrgSize] = useState("");
  const [saving, setSaving] = useState(false);
  const [legalPopup, setLegalPopup] = useState<"terms" | "privacy" | null>(null);

  const canNext0 = speciality && role && agreed;
  const canNext1 = orgName.trim();

  const handleFinish = async () => {
    if (!user) return;
    setSaving(true);
    await updateDoc(doc(db, "users", user.uid), {
      speciality,
      role,
      orgName: orgName.trim(),
      orgSize,
      onboarded: true,
      trialStartedAt: serverTimestamp(),
    });
    await refreshProfile();
    setSaving(false);
  };

  const dots = (
    <div style={{ display: "flex", justifyContent: "center", gap: 8, marginBottom: 24 }}>
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: i === step ? "var(--color-leaf)" : "var(--color-surface-inset)",
            transition: "background 0.2s",
          }}
        />
      ))}
    </div>
  );

  const legalPopupEl = legalPopup && (
    <div style={{
      position: "fixed",
      inset: 0,
      zIndex: 1200,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      background: "rgba(0,0,0,0.35)",
      backdropFilter: "blur(4px)",
      WebkitBackdropFilter: "blur(4px)",
    }}>
      <div style={{
        width: 500,
        maxWidth: "90vw",
        maxHeight: "70vh",
        background: "var(--color-bg)",
        borderRadius: 14,
        border: "1px solid var(--border-color)",
        boxShadow: "0 8px 40px rgba(0,0,0,0.18)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}>
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "16px 20px",
          borderBottom: "1px solid var(--border-color)",
          flexShrink: 0,
        }}>
          <h3 style={{
            fontFamily: "var(--font-display)",
            fontSize: 18,
            fontWeight: 400,
            color: "var(--color-ink)",
            margin: 0,
          }}>
            {legalPopup === "terms" ? "Terms and Conditions" : "Privacy Policy"}
          </h3>
          <button
            onClick={() => setLegalPopup(null)}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              color: "var(--color-ink-tertiary)",
              padding: 4,
              borderRadius: 6,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <X size={18} />
          </button>
        </div>
        <div style={{
          padding: "8px 20px 24px",
          overflowY: "auto",
          flex: 1,
        }}>
          {legalPopup === "terms" ? <TermsContent /> : <PrivacyContent />}
        </div>
      </div>
    </div>
  );

  if (step === 0) {
    return (
      <div>
        {legalPopupEl}
        {dots}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: 20 }}>
          <img
            src={quinnLogo}
            alt="Quinn logo"
            style={{ width: 28, height: 28, display: "block", marginBottom: 10 }}
          />
          <h2 style={{
            fontFamily: "var(--font-display)",
            fontSize: 22,
            fontWeight: 400,
            color: "var(--color-ink)",
            textAlign: "center",
            marginBottom: 0,
          }}>
            About you
          </h2>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <StyledSelect
            value={speciality}
            onValueChange={setSpeciality}
            placeholder="Specialty"
            options={SPECIALITIES}
          />

          <StyledSelect
            value={role}
            onValueChange={setRole}
            placeholder="Role"
            options={ROLES}
          />

          <div style={{
            display: "flex",
            alignItems: "flex-start",
            gap: 10,
            fontSize: 13,
            color: "var(--color-ink-secondary)",
            marginTop: 4,
          }}>
            <input
              type="checkbox"
              checked={agreed}
              onChange={(e) => setAgreed(e.target.checked)}
              style={{ marginTop: 2, accentColor: "var(--color-leaf)", cursor: "pointer" }}
            />
            <span>
              I agree to the{" "}
              <button
                type="button"
                onClick={() => setLegalPopup("terms")}
                style={{ background: "none", border: "none", padding: 0, color: "var(--color-leaf)", textDecoration: "underline", fontWeight: 500, cursor: "pointer", fontFamily: "var(--font-body)", fontSize: 13 }}
              >
                Terms
              </button>
              {" "}and the{" "}
              <button
                type="button"
                onClick={() => setLegalPopup("privacy")}
                style={{ background: "none", border: "none", padding: 0, color: "var(--color-leaf)", textDecoration: "underline", fontWeight: 500, cursor: "pointer", fontFamily: "var(--font-body)", fontSize: 13 }}
              >
                Privacy Policy
              </button>
            </span>
          </div>

          <button
            disabled={!canNext0}
            onClick={() => setStep(1)}
            style={{ ...primaryBtnStyle, opacity: canNext0 ? 1 : 0.5, cursor: canNext0 ? "pointer" : "not-allowed", marginTop: 8 }}
          >
            Continue
          </button>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div>
        {dots}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: 20 }}>
          <img
            src={quinnLogo}
            alt="Quinn logo"
            style={{ width: 28, height: 28, display: "block", marginBottom: 10 }}
          />
          <h2 style={{
            fontFamily: "var(--font-display)",
            fontSize: 22,
            fontWeight: 400,
            color: "var(--color-ink)",
            textAlign: "center",
            marginBottom: 0,
          }}>
            Your organization
          </h2>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <input
            style={inputStyle}
            placeholder="Organization name"
            value={orgName}
            onChange={(e) => setOrgName(e.target.value)}
            onFocus={(e) => e.currentTarget.style.borderColor = "var(--color-leaf)"}
            onBlur={(e) => e.currentTarget.style.borderColor = "var(--border-color)"}
          />

          <StyledSelect
            value={orgSize}
            onValueChange={setOrgSize}
            placeholder="How many dentists work at your clinic?"
            options={ORG_SIZES}
          />

          <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
            <button
              onClick={() => setStep(0)}
              style={{
                flex: 1,
                padding: "11px 0",
                background: "transparent",
                color: "var(--color-ink-secondary)",
                border: "1px solid var(--border-emphasis)",
                borderRadius: 8,
                fontSize: 14,
                fontWeight: 500,
                cursor: "pointer",
                fontFamily: "var(--font-body)",
              }}
            >
              Back
            </button>
            <button
              disabled={!canNext1}
              onClick={() => setStep(2)}
              style={{ ...primaryBtnStyle, flex: 2, opacity: canNext1 ? 1 : 0.5, cursor: canNext1 ? "pointer" : "not-allowed" }}
            >
              Continue
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Step 2 — Welcome
  return (
    <div>
      {dots}
      <div style={{ textAlign: "center" }}>
        <img
          src={quinnLogo}
          alt="Quinn logo"
          style={{ width: 28, height: 28, display: "block", margin: "0 auto 10px" }}
        />
        <h2 style={{
          fontFamily: "var(--font-display)",
          fontSize: 24,
          fontWeight: 400,
          color: "var(--color-ink)",
          marginBottom: 8,
        }}>
          Welcome!
        </h2>
        <p style={{
          fontSize: 14,
          color: "var(--color-ink-secondary)",
          marginBottom: 24,
          lineHeight: 1.6,
        }}>
          You have 7 days of free access on us.
        </p>
        <button
          onClick={handleFinish}
          disabled={saving}
          style={{ ...primaryBtnStyle, opacity: saving ? 0.7 : 1 }}
        >
          {saving ? "Saving..." : "Get Started"}
        </button>
      </div>
    </div>
  );
}
