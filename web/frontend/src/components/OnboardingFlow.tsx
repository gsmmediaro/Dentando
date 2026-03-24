import { useState } from "react";
import { doc, updateDoc, serverTimestamp } from "firebase/firestore";
import * as SelectPrimitive from "@radix-ui/react-select";
import { ChevronDown, X } from "lucide-react";
import { db } from "../firebase";
import { useAuth } from "../contexts/AuthContext";
import { useTranslation } from "react-i18next";

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

type TranslatedOption = { value: string; label: string };

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
  options: TranslatedOption[];
}) {
  const selectedLabel = options.find((o) => o.value === value)?.label;
  return (
    <SelectPrimitive.Root value={value || undefined} onValueChange={onValueChange}>
      <SelectPrimitive.Trigger style={triggerStyle(!!value)}>
        <SelectPrimitive.Value placeholder={placeholder}>
          {selectedLabel}
        </SelectPrimitive.Value>
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
                key={opt.value}
                value={opt.value}
                style={itemStyle}
                className="data-[highlighted]:bg-leaf-subtle"
              >
                <SelectPrimitive.ItemText>{opt.label}</SelectPrimitive.ItemText>
              </SelectPrimitive.Item>
            ))}
          </SelectPrimitive.Viewport>
        </SelectPrimitive.Content>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  );
}

function TermsContent() {
  const { t } = useTranslation();
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
      <h2 style={h2}>{t("onboarding.termsContent.s1h")}</h2>
      <p style={p}>{t("onboarding.termsContent.s1p")}</p>
      <h2 style={h2}>{t("onboarding.termsContent.s2h")}</h2>
      <p style={p}>{t("onboarding.termsContent.s2p")}</p>
      <h2 style={h2}>{t("onboarding.termsContent.s3h")}</h2>
      <p style={p}>{t("onboarding.termsContent.s3p")}</p>
      <h2 style={h2}>{t("onboarding.termsContent.s4h")}</h2>
      <p style={p}>{t("onboarding.termsContent.s4p")}</p>
      <h2 style={h2}>{t("onboarding.termsContent.s5h")}</h2>
      <p style={p}>{t("onboarding.termsContent.s5p")}</p>
      <h2 style={h2}>{t("onboarding.termsContent.s6h")}</h2>
      <p style={p}>{t("onboarding.termsContent.s6p")}</p>
      <h2 style={h2}>{t("onboarding.termsContent.s7h")}</h2>
      <p style={p}>{t("onboarding.termsContent.s7p")}</p>
    </>
  );
}

function PrivacyContent() {
  const { t } = useTranslation();
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
      <h2 style={h2}>{t("onboarding.privacyContent.s1h")}</h2>
      <p style={p}>{t("onboarding.privacyContent.s1p")}</p>
      <h2 style={h2}>{t("onboarding.privacyContent.s2h")}</h2>
      <p style={p}>{t("onboarding.privacyContent.s2p")}</p>
      <h2 style={h2}>{t("onboarding.privacyContent.s3h")}</h2>
      <p style={p}>{t("onboarding.privacyContent.s3p")}</p>
      <h2 style={h2}>{t("onboarding.privacyContent.s4h")}</h2>
      <p style={p}>{t("onboarding.privacyContent.s4p")}</p>
      <h2 style={h2}>{t("onboarding.privacyContent.s5h")}</h2>
      <p style={p}>{t("onboarding.privacyContent.s5p")}</p>
      <h2 style={h2}>{t("onboarding.privacyContent.s6h")}</h2>
      <p style={p}>{t("onboarding.privacyContent.s6p")}</p>
    </>
  );
}

export default function OnboardingFlow() {
  const { t } = useTranslation();
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
            {legalPopup === "terms" ? t("onboarding.legalModal.terms") : t("onboarding.legalModal.privacy")}
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
            src="/Cavio Logo.png"
            alt="Cavio logo"
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
            {t("onboarding.step0.title")}
          </h2>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <StyledSelect
            value={speciality}
            onValueChange={setSpeciality}
            placeholder={t("onboarding.step0.specialtyPlaceholder")}
            options={SPECIALITIES.map((s) => ({ value: s, label: t(`onboarding.specialties.${s}`, { defaultValue: s }) }))}
          />

          <StyledSelect
            value={role}
            onValueChange={setRole}
            placeholder={t("onboarding.step0.rolePlaceholder")}
            options={ROLES.map((r) => ({ value: r, label: t(`onboarding.roles.${r}`, { defaultValue: r }) }))}
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
              {t("onboarding.step0.agreeText")}{" "}
              <button
                type="button"
                onClick={() => setLegalPopup("terms")}
                style={{ background: "none", border: "none", padding: 0, color: "var(--color-leaf)", textDecoration: "underline", fontWeight: 500, cursor: "pointer", fontFamily: "var(--font-body)", fontSize: 13 }}
              >
                {t("onboarding.step0.terms")}
              </button>
              {" "}{t("onboarding.step0.and")}{" "}
              <button
                type="button"
                onClick={() => setLegalPopup("privacy")}
                style={{ background: "none", border: "none", padding: 0, color: "var(--color-leaf)", textDecoration: "underline", fontWeight: 500, cursor: "pointer", fontFamily: "var(--font-body)", fontSize: 13 }}
              >
                {t("onboarding.step0.privacy")}
              </button>
            </span>
          </div>

          <button
            disabled={!canNext0}
            onClick={() => setStep(1)}
            style={{ ...primaryBtnStyle, opacity: canNext0 ? 1 : 0.5, cursor: canNext0 ? "pointer" : "not-allowed", marginTop: 8 }}
          >
            {t("onboarding.continue")}
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
            src="/Cavio Logo.png"
            alt="Cavio logo"
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
            {t("onboarding.step1.title")}
          </h2>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <input
            style={inputStyle}
            placeholder={t("onboarding.step1.orgName")}
            value={orgName}
            onChange={(e) => setOrgName(e.target.value)}
            onFocus={(e) => e.currentTarget.style.borderColor = "var(--color-leaf)"}
            onBlur={(e) => e.currentTarget.style.borderColor = "var(--border-color)"}
          />

          <StyledSelect
            value={orgSize}
            onValueChange={setOrgSize}
            placeholder={t("onboarding.step1.dentistsPlaceholder")}
            options={ORG_SIZES.map((s) => ({ value: s, label: t(`onboarding.orgSizes.${s}`, { defaultValue: s }) }))}
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
              {t("onboarding.step1.back")}
            </button>
            <button
              disabled={!canNext1}
              onClick={() => setStep(2)}
              style={{ ...primaryBtnStyle, flex: 2, opacity: canNext1 ? 1 : 0.5, cursor: canNext1 ? "pointer" : "not-allowed" }}
            >
              {t("onboarding.continue")}
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
          src="/Cavio Logo.png"
          alt="Cavio logo"
          style={{ width: 28, height: 28, display: "block", margin: "0 auto 10px" }}
        />
        <h2 style={{
          fontFamily: "var(--font-display)",
          fontSize: 24,
          fontWeight: 400,
          color: "var(--color-ink)",
          marginBottom: 8,
        }}>
          {t("onboarding.step2.title")}
        </h2>
        <p style={{
          fontSize: 14,
          color: "var(--color-ink-secondary)",
          marginBottom: 24,
          lineHeight: 1.6,
        }}>
          {t("onboarding.step2.trialText")}
        </p>
        <button
          onClick={handleFinish}
          disabled={saving}
          style={{ ...primaryBtnStyle, opacity: saving ? 0.7 : 1 }}
        >
          {saving ? t("onboarding.step2.saving") : t("onboarding.step2.getStarted")}
        </button>
      </div>
    </div>
  );
}
