import { NavLink, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Plus, BarChart3, Users, LogOut, ChevronRight, ChevronLeft } from "lucide-react";
import { getPatientsFromFirestore, getPatientScansFromFirestore, type PatientSummary, type ScanRecord } from "../api/client";
import { useAuth } from "../contexts/AuthContext";
import quinnLogo from "../../../../quinnnlogo.svg";

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
  const { user, userProfile, logout } = useAuth();
  const [panelOpen, setPanelOpen] = useState(false);
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<string | null>(null);
  const [profileOpen, setProfileOpen] = useState(false);
  const profileRef = useRef<HTMLDivElement>(null);

  const togglePanel = () => {
    setPanelOpen(prev => !prev);
    if (panelOpen) setSelectedPatient(null);
  };

  useEffect(() => {
    if (panelOpen && user) {
      getPatientsFromFirestore(user.uid).then(setPatients);
    }
  }, [panelOpen, user]);


  const handlePatientClick = async (name: string) => {
    if (!user) return;
    setSelectedPatient(name);
    const scans = await getPatientScansFromFirestore(user.uid, name);
    onSelectPatient?.(scans, name);
  };

  const isDashboard = location.pathname === "/dashboard";

  const displayName = userProfile
    ? `${userProfile.firstName} ${userProfile.lastName}`.trim()
    : user?.displayName || user?.email || "";
  const userInitials = displayName
    ? initials(displayName)
    : (user?.email?.[0]?.toUpperCase() || "?");

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
          Quinn
        </div>

        {/* New session — dark green rectangle, off-white text */}
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
              Patients
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
