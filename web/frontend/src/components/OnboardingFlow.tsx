import { useState } from "react";
import { doc, updateDoc, serverTimestamp } from "firebase/firestore";
import * as SelectPrimitive from "@radix-ui/react-select";
import { ChevronDown, X } from "lucide-react";
import { db } from "../firebase";
import { useAuth } from "../contexts/AuthContext";

const SPECIALITIES = [
  "Dentist generalist",
  "Ortodont",
  "Endodont",
  "Chirurg oral",
  "Pedodont",
  "Periodontolog",
  "Prosthodont",
  "Altele",
];

const ROLES = ["Fondator", "Dentist", "Asistent", "Manager", "Altele"];
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
  const h2: React.CSSProperties = { fontSize: 15, fontWeight: 600, marginTop: 20, marginBottom: 6 };
  const p: React.CSSProperties = { fontSize: 13, color: "var(--color-ink-secondary)", marginBottom: 12, lineHeight: 1.65 };
  return (
    <>
      <h2 style={h2}>1. Acceptarea termenilor</h2>
      <p style={p}>Prin utilizarea platformei DentalAI, acceptati acesti termeni si conditii in totalitate. Daca nu sunteti de acord cu oricare dintre aceste prevederi, va rugam sa nu utilizati platforma.</p>
      <h2 style={h2}>2. Descrierea serviciului</h2>
      <p style={p}>DentalAI este un instrument de suport bazat pe inteligenta artificiala, conceput pentru a asista profesionistii din domeniul stomatologic in analiza radiografiilor dentare. Platforma nu inlocuieste diagnosticul clinic profesional.</p>
      <h2 style={h2}>3. Utilizarea corecta</h2>
      <p style={p}>Rezultatele furnizate de platforma sunt orientative si trebuie validate de un profesionist calificat. Utilizatorul este responsabil pentru toate deciziile clinice luate pe baza informatiilor furnizate de platforma.</p>
      <h2 style={h2}>4. Contul de utilizator</h2>
      <p style={p}>Sunteti responsabil pentru mentinerea confidentialitatii contului si parolei dumneavoastra. Orice activitate efectuata prin contul dumneavoastra este responsabilitatea dumneavoastra.</p>
      <h2 style={h2}>5. Perioada de proba</h2>
      <p style={p}>Noii utilizatori beneficiaza de o perioada de proba gratuita de 7 zile. Dupa expirarea perioadei de proba, accesul la functiile premium poate fi restrictionat.</p>
      <h2 style={h2}>6. Limitarea responsabilitatii</h2>
      <p style={p}>DentalAI nu isi asuma responsabilitatea pentru deciziile clinice luate pe baza rezultatelor platformei. Serviciul este furnizat "ca atare", fara garantii de niciun fel.</p>
      <h2 style={h2}>7. Modificari ale termenilor</h2>
      <p style={p}>Ne rezervam dreptul de a modifica acesti termeni in orice moment. Utilizatorii vor fi notificati prin intermediul platformei despre orice modificari semnificative.</p>
    </>
  );
}

function PrivacyContent() {
  const h2: React.CSSProperties = { fontSize: 15, fontWeight: 600, marginTop: 20, marginBottom: 6 };
  const p: React.CSSProperties = { fontSize: 13, color: "var(--color-ink-secondary)", marginBottom: 12, lineHeight: 1.65 };
  return (
    <>
      <h2 style={h2}>1. Datele colectate</h2>
      <p style={p}>Colectam urmatoarele date: nume, adresa de email, informatii despre organizatie, si datele radiografiilor incarcate pentru analiza. Aceste date sunt necesare pentru furnizarea serviciului.</p>
      <h2 style={h2}>2. Utilizarea datelor</h2>
      <p style={p}>Datele dumneavoastra sunt utilizate exclusiv pentru furnizarea si imbunatatirea serviciului DentalAI. Nu vindem si nu partajam datele personale cu terti fara consimtamantul dumneavoastra explicit.</p>
      <h2 style={h2}>3. Stocarea datelor</h2>
      <p style={p}>Datele sunt stocate securizat pe servere Firebase (Google Cloud). Fiecare utilizator are acces doar la propriile date. Implementam masuri de securitate adecvate pentru protejarea informatiilor.</p>
      <h2 style={h2}>4. Datele pacientilor</h2>
      <p style={p}>Radiografiile si datele pacientilor sunt stocate criptat si sunt accesibile doar utilizatorului care le-a incarcat. Nu utilizam datele pacientilor pentru antrenarea modelelor AI fara consimtamant explicit.</p>
      <h2 style={h2}>5. Drepturile utilizatorului</h2>
      <p style={p}>Aveti dreptul sa accesati, sa modificati sau sa stergeti datele dumneavoastra personale in orice moment. Pentru solicitari legate de datele personale, contactati-ne la adresa de email de suport.</p>
      <h2 style={h2}>6. Cookie-uri</h2>
      <p style={p}>Platforma utilizeaza cookie-uri esentiale pentru autentificare si functionarea corecta a serviciului. Nu utilizam cookie-uri de urmarire sau publicitate.</p>
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
            {legalPopup === "terms" ? "Termeni si conditii" : "Politica de confidentialitate"}
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
        <h2 style={{
          fontFamily: "var(--font-display)",
          fontSize: 22,
          fontWeight: 400,
          color: "var(--color-ink)",
          textAlign: "center",
          marginBottom: 20,
        }}>
          Despre tine
        </h2>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <StyledSelect
            value={speciality}
            onValueChange={setSpeciality}
            placeholder="Specialitate"
            options={SPECIALITIES}
          />

          <StyledSelect
            value={role}
            onValueChange={setRole}
            placeholder="Rol"
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
              Sunt de acord cu{" "}
              <button
                type="button"
                onClick={() => setLegalPopup("terms")}
                style={{ background: "none", border: "none", padding: 0, color: "var(--color-leaf)", textDecoration: "underline", fontWeight: 500, cursor: "pointer", fontFamily: "var(--font-body)", fontSize: 13 }}
              >
                Termenii
              </button>
              {" "}si{" "}
              <button
                type="button"
                onClick={() => setLegalPopup("privacy")}
                style={{ background: "none", border: "none", padding: 0, color: "var(--color-leaf)", textDecoration: "underline", fontWeight: 500, cursor: "pointer", fontFamily: "var(--font-body)", fontSize: 13 }}
              >
                Politica de confidentialitate
              </button>
            </span>
          </div>

          <button
            disabled={!canNext0}
            onClick={() => setStep(1)}
            style={{ ...primaryBtnStyle, opacity: canNext0 ? 1 : 0.5, cursor: canNext0 ? "pointer" : "not-allowed", marginTop: 8 }}
          >
            Continua
          </button>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div>
        {dots}
        <h2 style={{
          fontFamily: "var(--font-display)",
          fontSize: 22,
          fontWeight: 400,
          color: "var(--color-ink)",
          textAlign: "center",
          marginBottom: 20,
        }}>
          Organizatia ta
        </h2>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <input
            style={inputStyle}
            placeholder="Numele organizatiei"
            value={orgName}
            onChange={(e) => setOrgName(e.target.value)}
            onFocus={(e) => e.currentTarget.style.borderColor = "var(--color-leaf)"}
            onBlur={(e) => e.currentTarget.style.borderColor = "var(--border-color)"}
          />

          <StyledSelect
            value={orgSize}
            onValueChange={setOrgSize}
            placeholder="Cati dentisti lucreaza in cabinet?"
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
              Inapoi
            </button>
            <button
              disabled={!canNext1}
              onClick={() => setStep(2)}
              style={{ ...primaryBtnStyle, flex: 2, opacity: canNext1 ? 1 : 0.5, cursor: canNext1 ? "pointer" : "not-allowed" }}
            >
              Continua
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
        <div style={{ fontSize: 48, marginBottom: 12 }}>🎉</div>
        <h2 style={{
          fontFamily: "var(--font-display)",
          fontSize: 24,
          fontWeight: 400,
          color: "var(--color-ink)",
          marginBottom: 8,
        }}>
          Bun venit!
        </h2>
        <p style={{
          fontSize: 14,
          color: "var(--color-ink-secondary)",
          marginBottom: 24,
          lineHeight: 1.6,
        }}>
          Ai 7 zile de acces gratuit, din partea noastra!
        </p>
        <button
          onClick={handleFinish}
          disabled={saving}
          style={{ ...primaryBtnStyle, opacity: saving ? 0.7 : 1 }}
        >
          {saving ? "Se salveaza..." : "Incepe"}
        </button>
      </div>
    </div>
  );
}
