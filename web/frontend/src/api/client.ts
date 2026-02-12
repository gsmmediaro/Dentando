import {
  collection,
  addDoc,
  getDocs,
  getDoc,
  setDoc,
  doc,
  query,
  orderBy,
  limit as fsLimit,
  where,
  serverTimestamp,
  Timestamp,
} from "firebase/firestore";
import { db } from "../firebase";

const BASE = "";

export interface Detection {
  class_name: string;
  confidence: number;
  bbox: number[];
}

export interface AnalysisResult {
  filename: string;
  suspicion_level: string;
  overall_confidence: number;
  detections: Detection[];
  annotated_image_url: string;
  modality: string;
  model_name: string;
  num_detections: number;
  turnaround_s: number;
}

export interface ScanRecord {
  id: string;
  timestamp: string;
  filename: string;
  patient_name: string;
  suspicion: string;
  confidence: number;
  detections_count: number;
  modality: string;
  turnaround_s: number;
}

export interface PatientSummary {
  name: string;
  scan_count: number;
  last_scan: string;
  worst_suspicion: string;
}

export interface DailyStats {
  total: number;
  high: number;
  review: number;
  avg_turnaround: number;
}

export interface ModelInfo {
  path: string;
  name: string;
}

/* ── Backend API calls (inference stays server-side) ── */

export async function analyzeImage(
  file: File,
  modelPath: string,
  confThreshold: number,
  modality: string,
  useToothAssignment: boolean,
  patientName: string,
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_path", modelPath);
  form.append("conf_threshold", String(confThreshold));
  form.append("modality", modality);
  form.append("use_tooth_assignment", String(useToothAssignment));
  form.append("patient_name", patientName);
  const res = await fetch(`${BASE}/api/analyze`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${BASE}/api/models`);
  return res.json();
}

/* ── Firestore CRUD (scoped to authenticated user) ── */

const SUSPICION_ORDER: Record<string, number> = { LOW: 0, MODERATE: 1, HIGH: 2, REVIEW: 3 };

export async function saveScanToFirestore(
  uid: string,
  scan: {
    filename: string;
    patientName: string;
    suspicion: string;
    confidence: number;
    detectionsCount: number;
    modality: string;
    turnaroundS: number;
  },
) {
  const scansRef = collection(db, "users", uid, "scans");
  await addDoc(scansRef, {
    timestamp: serverTimestamp(),
    filename: scan.filename,
    patientName: scan.patientName,
    suspicion: scan.suspicion,
    confidence: scan.confidence,
    detectionsCount: scan.detectionsCount,
    modality: scan.modality,
    turnaroundS: scan.turnaroundS,
  });

  // Update patient summary if patient name provided
  if (scan.patientName.trim()) {
    const patRef = doc(db, "users", uid, "patients", scan.patientName.trim());
    const patSnap = await getDoc(patRef);
    if (patSnap.exists()) {
      const data = patSnap.data();
      const oldWorst = SUSPICION_ORDER[data.worstSuspicion] ?? -1;
      const newWorst = SUSPICION_ORDER[scan.suspicion] ?? -1;
      await setDoc(patRef, {
        scanCount: (data.scanCount || 0) + 1,
        lastScan: new Date().toISOString(),
        worstSuspicion: newWorst > oldWorst ? scan.suspicion : data.worstSuspicion,
      });
    } else {
      await setDoc(patRef, {
        scanCount: 1,
        lastScan: new Date().toISOString(),
        worstSuspicion: scan.suspicion,
      });
    }
  }
}

export async function getHistoryFromFirestore(uid: string, count = 50): Promise<ScanRecord[]> {
  const q = query(
    collection(db, "users", uid, "scans"),
    orderBy("timestamp", "desc"),
    fsLimit(count),
  );
  const snap = await getDocs(q);
  return snap.docs.map((d) => {
    const data = d.data();
    const ts = data.timestamp instanceof Timestamp
      ? data.timestamp.toDate().toISOString()
      : new Date().toISOString();
    return {
      id: d.id,
      timestamp: ts,
      filename: data.filename || "",
      patient_name: data.patientName || "",
      suspicion: data.suspicion || "LOW",
      confidence: data.confidence || 0,
      detections_count: data.detectionsCount || 0,
      modality: data.modality || "",
      turnaround_s: data.turnaroundS || 0,
    };
  });
}

export async function getStatsFromFirestore(uid: string): Promise<DailyStats> {
  const todayStart = new Date();
  todayStart.setHours(0, 0, 0, 0);

  const q = query(
    collection(db, "users", uid, "scans"),
    where("timestamp", ">=", Timestamp.fromDate(todayStart)),
  );
  const snap = await getDocs(q);

  let total = 0;
  let high = 0;
  let review = 0;
  let sumTurnaround = 0;

  snap.docs.forEach((d) => {
    const data = d.data();
    total++;
    if (data.suspicion === "HIGH") high++;
    if (data.suspicion === "REVIEW") review++;
    sumTurnaround += data.turnaroundS || 0;
  });

  return {
    total,
    high,
    review,
    avg_turnaround: total > 0 ? Math.round((sumTurnaround / total) * 100) / 100 : 0,
  };
}

export async function getPatientsFromFirestore(uid: string): Promise<PatientSummary[]> {
  const snap = await getDocs(collection(db, "users", uid, "patients"));
  return snap.docs.map((d) => {
    const data = d.data();
    return {
      name: d.id,
      scan_count: data.scanCount || 0,
      last_scan: data.lastScan || "",
      worst_suspicion: data.worstSuspicion || "LOW",
    };
  });
}

export async function getPatientScansFromFirestore(uid: string, name: string): Promise<ScanRecord[]> {
  const q = query(
    collection(db, "users", uid, "scans"),
    where("patientName", "==", name),
    orderBy("timestamp", "desc"),
  );
  const snap = await getDocs(q);
  return snap.docs.map((d) => {
    const data = d.data();
    const ts = data.timestamp instanceof Timestamp
      ? data.timestamp.toDate().toISOString()
      : new Date().toISOString();
    return {
      id: d.id,
      timestamp: ts,
      filename: data.filename || "",
      patient_name: data.patientName || "",
      suspicion: data.suspicion || "LOW",
      confidence: data.confidence || 0,
      detections_count: data.detectionsCount || 0,
      modality: data.modality || "",
      turnaround_s: data.turnaroundS || 0,
    };
  });
}
