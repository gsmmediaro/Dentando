import { NavLink, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Plus, BarChart3, Users, LogOut, ChevronRight, ChevronLeft, Menu, X, ArrowLeft } from "lucide-react";
import { getPatientsFromFirestore, getPatientScansFromFirestore, type PatientSummary, type ScanRecord } from "../api/client";
import { useAuth } from "../contexts/AuthContext";
import quinnLogo from "../../../../quinnnlogo.svg";

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
  onSelectPatient?: (scans: ScanRecord[], name: string) => void;
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

export default function Layout({ children, onSelectPatient }: Props) {
  const location = useLocation();
  const isMobile = useIsMobile();
  const { user, userProfile, logout } = useAuth();
  const [panelOpen, setPanelOpen] = useState(false);
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<string | null>(null);
  const [profileOpen, setProfileOpen] = useState(false);
  const profileRef = useRef<HTMLDivElement>(null);

  // Mobile drawer state
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerView, setDrawerView] = useState<"menu" | "patients">("menu");

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
      getPatientsFromFirestore(user.uid).then(setPatients);
    } else if (!isMobile && panelOpen && user) {
      getPatientsFromFirestore(user.uid).then(setPatients);
    }
  }, [panelOpen, user, isMobile, drawerOpen, drawerView]);

  const handlePatientClick = async (name: string) => {
    if (!user) return;
    setSelectedPatient(name);
    const scans = await getPatientScansFromFirestore(user.uid, name);
    onSelectPatient?.(scans, name);
    if (isMobile) setDrawerOpen(false);
  };

  // Close mobile drawer on route change
  useEffect(() => {
    if (isMobile) {
      setDrawerOpen(false);
      setDrawerView("menu");
    }
  }, [location.pathname, isMobile]);

  const isDashboard = location.pathname === "/dashboard";

  const displayName = userProfile
    ? `${userProfile.firstName} ${userProfile.lastName}`.trim()
    : user?.displayName || user?.email || "";
  const userInitials = displayName
    ? initials(displayName)
    : (user?.email?.[0]?.toUpperCase() || "?");

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
            <img src={quinnLogo} alt="Quinn" style={{ width: 24, height: 24 }} />
            <span style={{ fontFamily: "var(--font-display)", fontSize: 18, fontWeight: 500, color: "var(--color-ink)" }}>Quinn</span>
          </div>
          <div style={{ fontSize: 11, color: "var(--color-ink-tertiary)", textAlign: "center", flex: 1, padding: "0 8px" }}>
            AI decision support tool
          </div>
        </div>

        {/* Main content */}
        <div style={{ flex: 1, paddingBottom: 72 }}>
          {children}
        </div>

        {/* Bottom nav bar */}
        <div style={{
          position: "fixed",
          bottom: 0,
          left: 0,
          right: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-around",
          padding: "8px 0 calc(8px + env(safe-area-inset-bottom, 0px))",
          background: "var(--color-surface)",
          borderTop: "1px solid var(--border-color)",
          zIndex: 25,
        }}>
          <NavLink to="/analyze" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2, textDecoration: "none", padding: "4px 12px" }}>
            <Plus size={20} style={{ color: location.pathname === "/analyze" ? "var(--color-leaf)" : "var(--color-ink-tertiary)" }} />
            <span style={{ fontSize: 10, fontWeight: 500, color: location.pathname === "/analyze" ? "var(--color-leaf)" : "var(--color-ink-tertiary)", fontFamily: "var(--font-body)" }}>Scan</span>
          </NavLink>
          <button
            onClick={() => { setDrawerView("patients"); setDrawerOpen(true); }}
            style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2, background: "none", border: "none", cursor: "pointer", padding: "4px 12px" }}
          >
            <Users size={20} style={{ color: "var(--color-ink-tertiary)" }} />
            <span style={{ fontSize: 10, fontWeight: 500, color: "var(--color-ink-tertiary)", fontFamily: "var(--font-body)" }}>Patients</span>
          </button>
          <NavLink to="/dashboard" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2, textDecoration: "none", padding: "4px 12px" }}>
            <BarChart3 size={20} style={{ color: isDashboard ? "var(--color-leaf)" : "var(--color-ink-tertiary)" }} />
            <span style={{ fontSize: 10, fontWeight: 500, color: isDashboard ? "var(--color-leaf)" : "var(--color-ink-tertiary)", fontFamily: "var(--font-body)" }}>Analytics</span>
          </NavLink>
          <button
            onClick={() => { setDrawerView("menu"); setDrawerOpen(true); }}
            style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2, background: "none", border: "none", cursor: "pointer", padding: "4px 12px" }}
          >
            <Menu size={20} style={{ color: "var(--color-ink-tertiary)" }} />
            <span style={{ fontSize: 10, fontWeight: 500, color: "var(--color-ink-tertiary)", fontFamily: "var(--font-body)" }}>More</span>
          </button>
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
                      <span style={{ fontFamily: "var(--font-display)", fontSize: 18, fontWeight: 500 }}>Menu</span>
                      <button
                        onClick={() => { setDrawerOpen(false); setDrawerView("menu"); }}
                        style={{ background: "none", border: "none", cursor: "pointer", padding: 4, color: "var(--color-ink-tertiary)" }}
                      >
                        <X size={20} />
                      </button>
                    </div>

                    <nav style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <NavLink
                        to="/analyze"
                        onClick={() => setDrawerOpen(false)}
                        style={{
                          ...linkBase,
                          color: "#faf9f6",
                          background: "var(--color-leaf)",
                          borderRadius: 8,
                          marginBottom: 8,
                        }}
                      >
                        <Plus size={16} />
                        New scan
                      </NavLink>

                      <button
                        style={{
                          ...linkBase,
                          color: "var(--color-ink-secondary)",
                          background: "transparent",
                        }}
                        onClick={() => setDrawerView("patients")}
                      >
                        <Users size={16} />
                        <span style={{ flex: 1 }}>Patients</span>
                        <ChevronRight size={14} style={{ color: "var(--color-ink-tertiary)" }} />
                      </button>

                      <NavLink
                        to="/dashboard"
                        onClick={() => setDrawerOpen(false)}
                        style={{
                          ...linkBase,
                          color: isDashboard ? "var(--color-ink)" : "var(--color-ink-secondary)",
                          background: isDashboard ? "var(--color-surface-hover)" : "transparent",
                        }}
                      >
                        <BarChart3 size={16} />
                        Analytics
                      </NavLink>

                      <NavLink
                        to="/history"
                        onClick={() => setDrawerOpen(false)}
                        style={{
                          ...linkBase,
                          color: location.pathname === "/history" ? "var(--color-ink)" : "var(--color-ink-secondary)",
                          background: location.pathname === "/history" ? "var(--color-surface-hover)" : "transparent",
                        }}
                      >
                        <BarChart3 size={16} />
                        History
                      </NavLink>
                    </nav>

                    {/* Profile section */}
                    {user && (
                      <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid var(--border-color)" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12, padding: "0 12px" }}>
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
                          <span style={{ fontSize: 13, fontWeight: 500, color: "var(--color-ink)" }}>
                            {displayName || user.email}
                          </span>
                        </div>
                        <button
                          onClick={() => { setDrawerOpen(false); logout(); }}
                          style={{
                            ...linkBase,
                            color: "var(--color-ink-secondary)",
                            background: "transparent",
                          }}
                        >
                          <LogOut size={14} />
                          Sign out
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
                      <span style={{ fontFamily: "var(--font-display)", fontSize: 18, fontWeight: 500, flex: 1 }}>Patients</span>
                      <button
                        onClick={() => { setDrawerOpen(false); setDrawerView("menu"); }}
                        style={{ background: "none", border: "none", cursor: "pointer", padding: 4, color: "var(--color-ink-tertiary)" }}
                      >
                        <X size={20} />
                      </button>
                    </div>

                    {patients.length === 0 ? (
                      <div style={{ textAlign: "center", padding: "32px 20px", color: "var(--color-ink-tertiary)" }}>
                        <div style={{ fontSize: 13 }}>No patients yet.</div>
                        <div style={{ fontSize: 12, marginTop: 4 }}>Scans with names will appear here.</div>
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
                              {p.scan_count} {p.scan_count === 1 ? "scan" : "scans"}
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
  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      {/* Sidebar */}
      <aside style={{
        width: "var(--sidebar-w)",
        background: "var(--color-bg)",
        borderRight: "1px solid var(--border-color)",
        padding: "20px 12px",
        display: "flex",
        flexDirection: "column",
        position: "fixed",
        top: 0,
        left: 0,
        bottom: 0,
        zIndex: 20,
      }}>
        {/* Brand */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "8px 12px",
          marginBottom: 20,
          fontFamily: "var(--font-display)",
          fontSize: 20,
          fontWeight: 500,
          color: "var(--color-ink)",
        }}>
          <img
            src={quinnLogo}
            alt="Quinn logo"
            style={{
              width: 28,
              height: 28,
              display: "block",
              flexShrink: 0,
            }}
          />
          <span style={{ fontFamily: "var(--font-display)" }}>Quinn</span>
        </div>

        {/* New session */}
        <NavLink
          to="/analyze"
          style={{
            ...linkBase,
            color: "#faf9f6",
            background: "var(--color-leaf)",
            borderRadius: 8,
          }}
          onMouseEnter={(e) => e.currentTarget.style.background = "var(--color-leaf-hover)"}
          onMouseLeave={(e) => e.currentTarget.style.background = "var(--color-leaf)"}
        >
          <Plus size={16} />
          New scan
        </NavLink>

        {/* Nav */}
        <nav style={{ display: "flex", flexDirection: "column", gap: 1, marginTop: 16 }}>
          <button
            style={{
              ...linkBase,
              color: panelOpen ? "var(--color-ink)" : "var(--color-ink-secondary)",
              background: panelOpen ? "var(--color-surface)" : "transparent",
            }}
            onMouseEnter={(e) => { if (!panelOpen) e.currentTarget.style.background = "var(--color-surface-hover)"; }}
            onMouseLeave={(e) => { if (!panelOpen) e.currentTarget.style.background = "transparent"; }}
            onClick={togglePanel}
          >
            <Users size={16} />
            <span style={{ flex: 1 }}>Patients</span>
            {panelOpen
              ? <ChevronLeft size={14} style={{ color: "var(--color-ink-tertiary)" }} />
              : <ChevronRight size={14} style={{ color: "var(--color-ink-tertiary)" }} />
            }
          </button>
          <NavLink
            to="/dashboard"
            end
            style={{
              ...linkBase,
              color: isDashboard ? "var(--color-ink)" : "var(--color-ink-secondary)",
              background: isDashboard ? "var(--color-surface)" : "transparent",
            }}
            onMouseEnter={(e) => { if (!isDashboard) e.currentTarget.style.background = "var(--color-surface-hover)"; }}
            onMouseLeave={(e) => { if (!isDashboard) e.currentTarget.style.background = "transparent"; }}
          >
            <BarChart3 size={16} />
            Analytics
          </NavLink>
        </nav>

        {/* Profile — at very bottom */}
        {user && (
          <div ref={profileRef} style={{ marginTop: "auto", position: "relative" }}>
            <button
              onClick={() => setProfileOpen(!profileOpen)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "10px 12px",
                borderRadius: 8,
                width: "100%",
                border: "none",
                cursor: "pointer",
                background: profileOpen ? "var(--color-surface)" : "transparent",
                transition: "background 0.15s",
                fontFamily: "var(--font-body)",
                textAlign: "left",
              }}
              onMouseEnter={(e) => { if (!profileOpen) e.currentTarget.style.background = "var(--color-surface-hover)"; }}
              onMouseLeave={(e) => { if (!profileOpen) e.currentTarget.style.background = profileOpen ? "var(--color-surface)" : "transparent"; }}
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
              <div style={{ minWidth: 0, flex: 1 }}>
                <div style={{
                  fontSize: 13,
                  fontWeight: 500,
                  color: "var(--color-ink)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}>
                  {displayName || user.email}
                </div>
              </div>
            </button>

            {/* Dropdown */}
            {profileOpen && (
              <>
                <div
                  style={{ position: "fixed", inset: 0, zIndex: 30 }}
                  onClick={() => setProfileOpen(false)}
                />
                <div style={{
                  position: "absolute",
                  bottom: "calc(100% + 6px)",
                  left: 0,
                  right: 0,
                  background: "var(--color-surface)",
                  border: "1px solid var(--border-emphasis)",
                  borderRadius: 10,
                  boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
                  padding: 4,
                  zIndex: 31,
                }}>
                  <button
                    onClick={() => { setProfileOpen(false); logout(); }}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      padding: "8px 12px",
                      width: "100%",
                      border: "none",
                      borderRadius: 6,
                      cursor: "pointer",
                      background: "transparent",
                      color: "var(--color-ink-secondary)",
                      fontSize: 13,
                      fontWeight: 500,
                      fontFamily: "var(--font-body)",
                      transition: "background 0.1s",
                      textAlign: "left",
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = "var(--color-surface-hover)"}
                    onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                  >
                    <LogOut size={14} />
                    Sign out
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </aside>

      {/* Patients panel */}
      <AnimatePresence>
        {panelOpen && (
          <motion.div
            style={{
              position: "fixed",
              top: 0,
              left: "var(--sidebar-w)",
              bottom: 0,
              width: "var(--panel-w)",
              background: "var(--color-surface)",
              borderRight: "1px solid var(--border-color)",
              zIndex: 15,
              overflowY: "auto",
              padding: "20px 0",
            }}
            initial={{ x: -280, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -280, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          >
            <div style={{
              padding: "8px 20px 16px",
              fontFamily: "var(--font-display)",
              fontSize: 18,
              fontWeight: 500,
              borderBottom: "1px solid var(--border-color)",
              marginBottom: 12,
            }}>
              <span style={{ fontFamily: "var(--font-display)" }}>Patients</span>
            </div>
            {patients.length === 0 ? (
              <div style={{ textAlign: "center", padding: "32px 20px", color: "var(--color-ink-tertiary)" }}>
                <div style={{ fontSize: 13 }}>No patients yet.</div>
                <div style={{ fontSize: 12, marginTop: 4 }}>Scans with names will appear here.</div>
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
                    padding: "10px 20px",
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
                    width: 32,
                    height: 32,
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
                      {p.scan_count} {p.scan_count === 1 ? "scan" : "scans"}
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
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main content */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          minHeight: "100vh",
          position: "absolute",
          top: 0,
          right: 0,
          bottom: 0,
          left: panelOpen ? "calc(var(--sidebar-w) + var(--panel-w))" : "var(--sidebar-w)",
        }}
      >
        <div style={{
          textAlign: "center",
          padding: "8px 16px",
          fontSize: 12,
          color: "var(--color-ink-tertiary)",
          borderBottom: "1px solid var(--border-color)",
        }}>
          AI decision support tool &mdash; does not replace clinical diagnosis
        </div>
        {children}
      </div>
    </div>
  );
}
