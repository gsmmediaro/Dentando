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
        Politica de confidentialitate
      </h1>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>1. Datele colectate</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        Colectam urmatoarele date: nume, adresa de email, informatii despre organizatie, si datele radiografiilor incarcate pentru analiza. Aceste date sunt necesare pentru furnizarea serviciului.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>2. Utilizarea datelor</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        Datele dumneavoastra sunt utilizate exclusiv pentru furnizarea si imbunatatirea serviciului DentalAI. Nu vindem si nu partajam datele personale cu terti fara consimtamantul dumneavoastra explicit.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>3. Stocarea datelor</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        Datele sunt stocate securizat pe servere Firebase (Google Cloud). Fiecare utilizator are acces doar la propriile date. Implementam masuri de securitate adecvate pentru protejarea informatiilor.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>4. Datele pacientilor</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        Radiografiile si datele pacientilor sunt stocate criptat si sunt accesibile doar utilizatorului care le-a incarcat. Nu utilizam datele pacientilor pentru antrenarea modelelor AI fara consimtamant explicit.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>5. Drepturile utilizatorului</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        Aveti dreptul sa accesati, sa modificati sau sa stergeti datele dumneavoastra personale in orice moment. Pentru solicitari legate de datele personale, contactati-ne la adresa de email de suport.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>6. Cookie-uri</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        Platforma utilizeaza cookie-uri esentiale pentru autentificare si functionarea corecta a serviciului. Nu utilizam cookie-uri de urmarire sau publicitate.
      </p>

      <h2 style={{ fontSize: 16, fontWeight: 600, marginTop: 24, marginBottom: 8 }}>7. Contactul</h2>
      <p style={{ fontSize: 14, color: "var(--color-ink-secondary)", marginBottom: 16 }}>
        Pentru orice intrebari legate de aceasta politica de confidentialitate sau de datele dumneavoastra, nu ezitati sa ne contactati.
      </p>
    </div>
  );
}
