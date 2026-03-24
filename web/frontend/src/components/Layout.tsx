import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { Plus, Users, LogOut, ChevronRight, Menu, X, ArrowLeft, MessageCircle, HelpCircle, Settings, CornerDownRight } from "lucide-react";
import { getPatientsFromFirestore, type PatientSummary } from "../api/client";
import { useAuth } from "../contexts/AuthContext";
import { useTranslation } from "react-i18next";

function useIsMobile(breakpoint = 768) {
  const [mobile, setMobile] = useState(() => window.innerWidth <= breakpoint);
  useEffect(() => {
    const mq = window.matchMedia(`(max-width: ${breakpoint}px)`);
    const handler = (e: MediaQueryListEvent) => setMobile(e.matches);
    mq.addEventListener("change", handler);
    setMobile(mq.matches);
    return () => mq.removeEventListener("change", handler);
  }, [breakpoint]);
  return mobile;
}

function suspicionDotColor(s: string): string {
  switch (s) {
    case "HIGH": return "var(--color-high)";
    case "MODERATE": return "var(--color-moderate)";
    case "REVIEW": return "var(--color-review)";
    default: return "var(--color-low)";
  }
}

function initials(name: string): string {
  return name.split(" ").map(w => w[0]).join("").toUpperCase().slice(0, 2);
}

interface Props {
  children: ReactNode;
}

const linkBase: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 12,
  padding: "10px 12px",
  borderRadius: 6,
  fontSize: 14,
  fontWeight: 500,
  transition: "background 0.15s, color 0.15s",
  textDecoration: "none",
  cursor: "pointer",
  border: "none",
  width: "100%",
  textAlign: "left",
  fontFamily: "var(--font-body)",
};

export default function Layout({ children }: Props) {
  const { t } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  const isMobile = useIsMobile();
  const { user, userProfile, logout, setShowAuthGate } = useAuth();
  const [panelOpen, setPanelOpen] = useState(false);
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<string | null>(null);
  const [profileOpen, setProfileOpen] = useState(false);
  const profileRef = useRef<HTMLDivElement>(null);

  // Mobile drawer state
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerView, setDrawerView] = useState<"menu" | "patients">("menu");
  const [mobileProfileOpen, setMobileProfileOpen] = useState(false);
  const [settingsPopover, setSettingsPopover] = useState(false);

  const refreshPatients = useCallback(() => {
    if (!user) return;
    getPatientsFromFirestore(user.uid).then(setPatients);
  }, [user]);

  const togglePanel = () => {
    if (isMobile) {
      setDrawerView("patients");
      setDrawerOpen(true);
    } else {
      setPanelOpen(prev => !prev);
      if (panelOpen) setSelectedPatient(null);
    }
  };

  useEffect(() => {
    if (isMobile && drawerView === "patients" && drawerOpen && user) {
      refreshPatients();
    } else if (!isMobile && panelOpen && user) {
      refreshPatients();
    }
  }, [panelOpen, user, isMobile, drawerOpen, drawerView, refreshPatients]);

  useEffect(() => {
    const handler = () => refreshPatients();
    window.addEventListener("cavio:patients-updated", handler);
    return () => window.removeEventListener("cavio:patients-updated", handler);
  }, [refreshPatients]);

  const handleNewScanClick = (event?: React.MouseEvent) => {
    event?.preventDefault();
    setSelectedPatient(null);
    navigate(`/analyze?new=${Date.now()}`);
    if (isMobile) {
      setDrawerOpen(false);
      setDrawerView("menu");
    }
  };

  const handlePatientClick = (name: string) => {
    setSelectedPatient(name);
    navigate(`/analyze?patient=${encodeURIComponent(name)}`);
    if (isMobile) {
      setDrawerOpen(false);
      setDrawerView("menu");
    }
  };

  // Close mobile drawer on route change
  useEffect(() => {
    if (isMobile) {
      setDrawerOpen(false);
      setDrawerView("menu");
    }
  }, [location.pathname, isMobile]);

  useEffect(() => {
    if (!drawerOpen) {
      setMobileProfileOpen(false);
    }
  }, [drawerOpen]);

  const displayName = userProfile
    ? `${userProfile.firstName} ${userProfile.lastName}`.trim()
    : user?.displayName || user?.email || "";
  const userInitials = displayName
    ? initials(displayName)
    : (user?.email?.[0]?.toUpperCase() || "?");

  /* ───────────────────── LOGGED-OUT LAYOUT ───────────────────── */
  if (!user) {
    return (
      <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
        {/* Minimal top bar */}
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: isMobile ? "12px 16px" : "12px 24px",
          background: "var(--color-bg)",
          position: "sticky",
          top: 0,
          zIndex: 10,
        }}>
          <div style={{ display: "flex", alignItems: "center" }}>
            <img src="/Cavio Header.png" alt="Cavio" style={{ height: 28 }} />
          </div>
          <button
            onClick={() => setShowAuthGate(true)}
            style={{
              padding: "8px 20px",
              borderRadius: 8,
              border: "none",
              background: "var(--color-ink)",
              color: "white",
              fontSize: 13,
              fontWeight: 500,
              cursor: "pointer",
              fontFamily: "var(--font-body)",
              transition: "opacity 0.15s",
            }}
            onMouseEnter={(e) => e.currentTarget.style.opacity = "0.88"}
            onMouseLeave={(e) => e.currentTarget.style.opacity = "1"}
          >
            {t("layout.nav.logIn")}
          </button>
        </div>
        {/* Main content */}
        <div style={{ flex: 1 }}>
          {children}
        </div>
      </div>
    );
  }

  /* ───────────────────── MOBILE LAYOUT ───────────────────── */
  if (isMobile) {
    return (
      <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
        {/* Top bar */}
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 16px",
          borderBottom: "1px solid var(--border-color)",
          background: "var(--color-bg)",
          position: "sticky",
          top: 0,
          zIndex: 10,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button
              onClick={() => { setDrawerView("menu"); setDrawerOpen(true); }}
              aria-label={t("layout.nav.openMenu")}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: "none",
                border: "none",
                cursor: "pointer",
                padding: 2,
                color: "var(--color-ink-secondary)",
              }}
            >
              <Menu size={20} />
            </button>
            <img src="/Cavio Header.png" alt="Cavio" style={{ height: 26 }} />
          </div>
          <div style={{ flex: 1 }} />
        </div>

        {/* Main content */}
        <div style={{ flex: 1 }}>
          {children}
        </div>

        {/* Bottom drawer overlay + sheet */}
        <AnimatePresence>
          {drawerOpen && (
            <>
              {/* Backdrop */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                onClick={() => { setDrawerOpen(false); setDrawerView("menu"); }}
                style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)", zIndex: 30 }}
              />
              {/* Drawer */}
              <motion.div
                initial={{ y: "100%" }}
                animate={{ y: 0 }}
                exit={{ y: "100%" }}
                transition={{ type: "spring", stiffness: 300, damping: 32 }}
                style={{
                  position: "fixed",
                  bottom: 0,
                  left: 0,
                  right: 0,
                  maxHeight: "80vh",
                  background: "var(--color-surface)",
                  borderRadius: "16px 16px 0 0",
                  zIndex: 35,
                  overflowY: "auto",
                  paddingBottom: "env(safe-area-inset-bottom, 0px)",
                }}
              >
                {/* Handle */}
                <div style={{ display: "flex", justifyContent: "center", padding: "10px 0 4px" }}>
                  <div style={{ width: 36, height: 4, borderRadius: 2, background: "var(--color-ink-ghost)" }} />
                </div>

                {drawerView === "menu" ? (
                  /* ── Menu view ── */
                  <div style={{ padding: "8px 16px 24px" }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                      <div style={{ display: "flex", alignItems: "center" }}>
                        <img src="/Cavio Header.png" alt="Cavio" style={{ height: 26 }} />
                      </div>
                      <button
                        onClick={() => { setDrawerOpen(false); setDrawerView("menu"); }}
                        style={{ background: "none", border: "none", cursor: "pointer", padding: 4, color: "var(--color-ink-tertiary)" }}
                      >
                        <X size={20} />
                      </button>
                    </div>

                    <nav style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <button
                        onClick={handleNewScanClick}
                        style={{
                          ...linkBase,
                          color: "var(--color-leaf)",
                          background: "transparent",
                          gap: 10,
                        }}
                      >
                        <div style={{ width: 18, height: 18, borderRadius: "50%", background: "var(--color-leaf)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                          <Plus size={11} strokeWidth={3} color="white" />
                        </div>
                        {t("layout.nav.newScan")}
                      </button>

                      <button
                        style={{
                          ...linkBase,
                          color: "var(--color-ink-secondary)",
                          background: "transparent",
                        }}
                        onClick={() => setDrawerView("patients")}
                      >
                        <Users size={16} />
                        <span style={{ flex: 1 }}>{t("layout.nav.recentScans")}</span>
                        <ChevronRight size={14} style={{ color: "var(--color-ink-tertiary)" }} />
                      </button>
                    </nav>

                    {/* Profile section */}
                    {user && (
                      <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid var(--border-color)", position: "relative" }}>
                        {mobileProfileOpen && (
                          <>
                            <div
                              style={{ position: "fixed", inset: 0, zIndex: 40 }}
                              onClick={() => setMobileProfileOpen(false)}
                            />
                            <div
                              style={{
                                position: "absolute",
                                left: 0,
                                right: 0,
                                bottom: "calc(100% + 6px)",
                                background: "var(--color-surface)",
                                border: "1px solid var(--border-emphasis)",
                                borderRadius: 10,
                                boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
                                padding: 4,
                                zIndex: 41,
                              }}
                            >
                              <button
                                onClick={() => { setDrawerOpen(false); setDrawerView("menu"); setMobileProfileOpen(false); logout(); }}
                                style={{
                                  ...linkBase,
                                  color: "var(--color-ink-secondary)",
                                  background: "transparent",
                                  padding: "8px 12px",
                                }}
                              >
                                <LogOut size={14} />
                                {t("layout.nav.signOut")}
                              </button>
                            </div>
                          </>
                        )}
                        <button
                          onClick={() => setMobileProfileOpen((prev) => !prev)}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 10,
                            width: "100%",
                            padding: "8px 12px",
                            borderRadius: 8,
                            border: "none",
                            cursor: "pointer",
                            background: mobileProfileOpen ? "var(--color-surface-hover)" : "transparent",
                            textAlign: "left",
                            fontFamily: "var(--font-body)",
                            position: "relative",
                            zIndex: 42,
                          }}
                        >
                          <div style={{
                            width: 30,
                            height: 30,
                            borderRadius: "50%",
                            background: "var(--color-leaf-subtle)",
                            color: "var(--color-leaf-text)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: 12,
                            fontWeight: 600,
                            flexShrink: 0,
                          }}>
                            {userInitials}
                          </div>
                          <span style={{ fontSize: 13, fontWeight: 500, color: "var(--color-ink)", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {displayName || user.email}
                          </span>
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  /* ── Patients view ── */
                  <div style={{ padding: "8px 0 24px" }}>
                    <div style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      padding: "0 16px 12px",
                      borderBottom: "1px solid var(--border-color)",
                      marginBottom: 8,
                    }}>
                      <button
                        onClick={() => setDrawerView("menu")}
                        style={{ background: "none", border: "none", cursor: "pointer", padding: 4, color: "var(--color-ink-secondary)", display: "flex", alignItems: "center" }}
                      >
                        <ArrowLeft size={20} />
                      </button>
                      <span style={{ fontFamily: "var(--font-display)", fontSize: 18, fontWeight: 500, flex: 1 }}>{t("layout.nav.patients")}</span>
                      <button
                        onClick={() => { setDrawerOpen(false); setDrawerView("menu"); }}
                        style={{ background: "none", border: "none", cursor: "pointer", padding: 4, color: "var(--color-ink-tertiary)" }}
                      >
                        <X size={20} />
                      </button>
                    </div>

                    {patients.length === 0 ? (
                      <div style={{ textAlign: "center", padding: "32px 20px", color: "var(--color-ink-tertiary)" }}>
                        <div style={{ fontSize: 13 }}>{t("layout.patients.noPatients")}</div>
                        <div style={{ fontSize: 12, marginTop: 4 }}>{t("layout.patients.scansWillAppear")}</div>
                      </div>
                    ) : (
                      patients.map((p) => (
                        <button
                          key={p.name}
                          onClick={() => handlePatientClick(p.name)}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 12,
                            padding: "12px 20px",
                            width: "100%",
                            textAlign: "left",
                            border: "none",
                            cursor: "pointer",
                            transition: "background 0.12s",
                            color: "var(--color-ink)",
                            fontFamily: "var(--font-body)",
                            fontSize: 14,
                            background: selectedPatient === p.name ? "var(--color-leaf-subtle)" : "transparent",
                          }}
                        >
                          <div style={{
                            width: 36,
                            height: 36,
                            borderRadius: "50%",
                            background: "var(--color-surface-inset)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: 13,
                            fontWeight: 600,
                            color: "var(--color-ink-secondary)",
                            flexShrink: 0,
                          }}>
                            {initials(p.name)}
                          </div>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontWeight: 500, lineHeight: 1.3 }}>{p.name}</div>
                            <div style={{ fontSize: 12, color: "var(--color-ink-tertiary)" }}>
                              {p.scan_count} {p.scan_count === 1 ? t("layout.patients.scan") : t("layout.patients.scans")}
                            </div>
                          </div>
                          <div style={{
                            width: 8,
                            height: 8,
                            borderRadius: "50%",
                            background: suspicionDotColor(p.worst_suspicion),
                            flexShrink: 0,
                          }} />
                        </button>
                      ))
                    )}
                  </div>
                )}
              </motion.div>
            </>
          )}
        </AnimatePresence>
      </div>
    );
  }

  /* ───────────────────── DESKTOP LAYOUT ───────────────────── */
  const sidebarOpen = panelOpen;
  const setSidebarOpen = setPanelOpen;

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      {/* Top bar */}
      <div style={{
        display: "flex",
        alignItems: "center",
        padding: "12px 24px",
        background: "var(--color-bg)",
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}>
        {/* Hamburger button */}
        <button
          onClick={() => { setSidebarOpen(true); refreshPatients(); }}
          aria-label={t("layout.nav.openMenu")}
          style={{
            width: 40,
            height: 40,
            borderRadius: "50%",
            border: "1px solid var(--border-color)",
            background: "var(--color-surface)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            color: "var(--color-ink-secondary)",
            transition: "border-color 0.15s, box-shadow 0.15s",
            flexShrink: 0,
          }}
          onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--border-emphasis)"; e.currentTarget.style.boxShadow = "0 1px 4px rgba(0,0,0,0.06)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border-color)"; e.currentTarget.style.boxShadow = "none"; }}
        >
          <Menu size={18} />
        </button>

        <div style={{ flex: 1 }} />

        {/* Profile avatar button — shown when logged in */}
        {user && (
          <div ref={profileRef} style={{ position: "relative" }}>
            <button
              onClick={() => setProfileOpen(prev => !prev)}
              style={{
                width: 36,
                height: 36,
                borderRadius: "50%",
                border: "none",
                background: "var(--color-leaf-subtle)",
                color: "var(--color-leaf-text)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
                fontFamily: "var(--font-body)",
                flexShrink: 0,
              }}
            >
              {userInitials}
            </button>
            {profileOpen && (
              <>
                <div style={{ position: "fixed", inset: 0, zIndex: 50 }} onClick={() => setProfileOpen(false)} />
                <div style={{
                  position: "absolute",
                  top: "calc(100% + 8px)",
                  right: 0,
                  background: "var(--color-surface)",
                  borderRadius: 14,
                  boxShadow: "0 4px 24px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.04)",
                  padding: 6,
                  minWidth: 180,
                  zIndex: 51,
                }}>
                  <button
                    onClick={() => { setProfileOpen(false); navigate("/settings"); }}
                    style={{
                      display: "flex", alignItems: "center", gap: 10,
                      padding: "11px 14px", width: "100%", border: "none",
                      borderRadius: 8, cursor: "pointer", background: "transparent",
                      color: "var(--color-ink)", fontSize: 14, fontWeight: 500,
                      fontFamily: "var(--font-body)", textAlign: "left",
                      transition: "background 0.1s, color 0.1s",
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "var(--color-surface-hover)"; e.currentTarget.style.color = "var(--color-leaf)"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--color-ink)"; }}
                  >
                    {t("layout.nav.accountSettings")}
                  </button>
                  <button
                    onClick={() => { setProfileOpen(false); logout(); }}
                    style={{
                      display: "flex", alignItems: "center", gap: 10,
                      padding: "11px 14px", width: "100%", border: "none",
                      borderRadius: 8, cursor: "pointer", background: "transparent",
                      color: "var(--color-ink)", fontSize: 14, fontWeight: 500,
                      fontFamily: "var(--font-body)", textAlign: "left",
                      transition: "background 0.1s, color 0.1s",
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "var(--color-surface-hover)"; e.currentTarget.style.color = "var(--color-leaf)"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--color-ink)"; }}
                  >
                    {t("layout.nav.logout")}
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Sidebar drawer overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={() => { setSidebarOpen(false); setSelectedPatient(null); }}
              style={{
                position: "fixed",
                inset: 0,
                background: "rgba(0,0,0,0.25)",
                backdropFilter: "blur(2px)",
                zIndex: 30,
              }}
            />
            {/* Sidebar panel */}
            <motion.aside
              initial={{ x: -320 }}
              animate={{ x: 0 }}
              exit={{ x: -320 }}
              transition={{ type: "spring", stiffness: 320, damping: 32 }}
              style={{
                position: "fixed",
                top: 0,
                left: 0,
                bottom: 0,
                width: 300,
                background: "var(--color-surface)",
                zIndex: 35,
                display: "flex",
                flexDirection: "column",
                boxShadow: "4px 0 24px rgba(0,0,0,0.08)",
              }}
            >
              {/* Header */}
              <div style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "20px 20px 16px",
              }}>
                <div style={{ display: "flex", alignItems: "center" }}>
                  <img src="/Cavio Header.png" alt="Cavio" style={{ height: 30 }} />
                </div>
                <button
                  onClick={() => { setSidebarOpen(false); setSelectedPatient(null); }}
                  style={{
                    width: 40, height: 40, borderRadius: "50%",
                    border: "1px solid var(--border-color)",
                    background: "var(--color-surface)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    cursor: "pointer", color: "var(--color-ink-secondary)",
                    transition: "border-color 0.15s, box-shadow 0.15s",
                    flexShrink: 0,
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--border-emphasis)"; e.currentTarget.style.boxShadow = "0 1px 4px rgba(0,0,0,0.06)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border-color)"; e.currentTarget.style.boxShadow = "none"; }}
                >
                  <X size={18} />
                </button>
              </div>

              {/* Navigation */}
              <nav style={{ display: "flex", flexDirection: "column", gap: 2, padding: "0 16px 0" }}>
                <button
                  onClick={(e) => { handleNewScanClick(e); setSidebarOpen(false); }}
                  style={{
                    ...linkBase,
                    color: "var(--color-leaf)",
                    background: "transparent",
                    gap: 10,
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = "var(--color-surface-hover)"; e.currentTarget.style.color = "var(--color-leaf)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--color-leaf)"; }}
                >
                  <div style={{ width: 18, height: 18, borderRadius: "50%", background: "var(--color-leaf)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    <Plus size={11} strokeWidth={3} color="white" />
                  </div>
                  {t("layout.nav.newScan")}
                </button>
              </nav>

              {/* Recent Patients section */}
              <div style={{
                flex: 1,
                overflowY: "auto",
                padding: "0 16px",
              }}>
                <div style={{
                  padding: "24px 12px 10px",
                  fontSize: 13,
                  fontWeight: 600,
                  color: "var(--color-ink)",
                }}>
                  {t("layout.nav.recentScans")}
                </div>
                {patients.length === 0 ? (
                  <div style={{ padding: "16px 12px", color: "var(--color-ink-tertiary)", fontSize: 13 }}>
                    {t("layout.patients.noScans")}
                  </div>
                ) : (
                  <>
                    {patients.slice(0, 5).map((p) => (
                      <button
                        key={p.name}
                        onClick={() => { handlePatientClick(p.name); setSidebarOpen(false); }}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 10,
                          padding: "9px 12px",
                          width: "100%",
                          textAlign: "left",
                          border: "none",
                          borderRadius: 6,
                          cursor: "pointer",
                          transition: "background 0.12s",
                          color: "var(--color-ink)",
                          fontFamily: "var(--font-body)",
                          fontSize: 14,
                          background: selectedPatient === p.name ? "var(--color-surface-hover)" : "transparent",
                        }}
                        onMouseEnter={(e) => { if (selectedPatient !== p.name) e.currentTarget.style.background = "var(--color-surface-hover)"; }}
                        onMouseLeave={(e) => { e.currentTarget.style.background = selectedPatient === p.name ? "var(--color-surface-hover)" : "transparent"; }}
                      >
                        <MessageCircle size={16} style={{ color: "var(--color-ink-tertiary)", flexShrink: 0 }} />
                        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{p.name}</span>
                      </button>
                    ))}
                    <button
                      onClick={() => { navigate("/history"); setSidebarOpen(false); }}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 10,
                        padding: "9px 12px",
                        width: "100%",
                        textAlign: "left",
                        border: "none",
                        borderRadius: 6,
                        cursor: "pointer",
                        transition: "background 0.12s",
                        color: "var(--color-ink-secondary)",
                        fontFamily: "var(--font-body)",
                        fontSize: 14,
                        background: "transparent",
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.background = "var(--color-surface-hover)"}
                      onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                    >
                      <CornerDownRight size={16} style={{ color: "var(--color-ink-tertiary)", flexShrink: 0 }} />
                      {t("layout.nav.viewAll")}
                    </button>
                  </>
                )}
              </div>

              {/* Bottom section */}
              <div style={{ padding: "0 16px 16px", display: "flex", flexDirection: "column", gap: 2 }}>
                <button
                  style={{
                    ...linkBase,
                    color: "var(--color-ink)",
                    background: "transparent",
                    gap: 10,
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = "var(--color-surface-hover)"; e.currentTarget.style.color = "var(--color-leaf)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--color-ink)"; }}
                >
                  <HelpCircle size={16} style={{ color: "var(--color-ink-secondary)" }} />
                  {t("layout.nav.helpResources")}
                </button>
                <div style={{ position: "relative" }}>
                  <button
                    onClick={() => setSettingsPopover(!settingsPopover)}
                    style={{
                      ...linkBase,
                      color: "var(--color-ink)",
                      background: settingsPopover ? "var(--color-surface-hover)" : "transparent",
                      gap: 10,
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "var(--color-surface-hover)"; e.currentTarget.style.color = "var(--color-leaf)"; }}
                    onMouseLeave={(e) => { if (!settingsPopover) e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--color-ink)"; }}
                  >
                    <Settings size={16} style={{ color: "var(--color-ink-secondary)" }} />
                    {t("layout.nav.settings")}
                  </button>
                  {settingsPopover && (
                    <>
                      <div
                        style={{ position: "fixed", inset: 0, zIndex: 40 }}
                        onClick={() => setSettingsPopover(false)}
                      />
                      <div style={{
                        position: "absolute",
                        bottom: "calc(100% + 6px)",
                        left: 0,
                        right: 0,
                        background: "var(--color-surface)",
                        borderRadius: 14,
                        boxShadow: "0 4px 24px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.04)",
                        padding: 6,
                        zIndex: 41,
                      }}>
                        <button
                          onClick={() => { setSettingsPopover(false); setSidebarOpen(false); navigate("/settings"); }}
                          style={{
                            display: "flex", alignItems: "center", gap: 10,
                            padding: "11px 14px", width: "100%", border: "none",
                            borderRadius: 8, cursor: "pointer", background: "transparent",
                            color: "var(--color-ink)", fontSize: 14, fontWeight: 500,
                            fontFamily: "var(--font-body)", textAlign: "left",
                            transition: "background 0.1s, color 0.1s",
                          }}
                          onMouseEnter={(e) => { e.currentTarget.style.background = "var(--color-surface-hover)"; e.currentTarget.style.color = "var(--color-leaf)"; }}
                          onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--color-ink)"; }}
                        >
                          {t("layout.nav.accountSettings")}
                        </button>
                        <button
                          onClick={() => { setSettingsPopover(false); setSidebarOpen(false); logout(); }}
                          style={{
                            display: "flex", alignItems: "center", gap: 10,
                            padding: "11px 14px", width: "100%", border: "none",
                            borderRadius: 8, cursor: "pointer", background: "transparent",
                            color: "var(--color-ink)", fontSize: 14, fontWeight: 500,
                            fontFamily: "var(--font-body)", textAlign: "left",
                            transition: "background 0.1s, color 0.1s",
                          }}
                          onMouseEnter={(e) => { e.currentTarget.style.background = "var(--color-surface-hover)"; e.currentTarget.style.color = "var(--color-leaf)"; }}
                          onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--color-ink)"; }}
                        >
                          {t("layout.nav.logout")}
                        </button>
                      </div>
                    </>
                  )}
                </div>

                <div style={{
                  padding: "12px 12px 0",
                  fontSize: 11,
                  lineHeight: 1.5,
                  color: "var(--color-ink-tertiary)",
                }}>
                  {t("layout.disclaimer")}{" "}
                  <a href="/terms" style={{ color: "var(--color-ink-secondary)", textDecoration: "underline" }}>{t("layout.footer.terms")}</a>{" "}{t("layout.footer.and")}{" "}
                  <a href="/privacy" style={{ color: "var(--color-ink-secondary)", textDecoration: "underline" }}>{t("layout.footer.privacy")}</a>.
                </div>
                <div style={{ padding: "6px 12px 0", fontSize: 11, color: "var(--color-ink-ghost)" }}>
                  v1.0.0
                </div>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Main content — full width */}
      <div style={{ flex: 1 }}>
        {children}
      </div>
    </div>
  );
}
