import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { Upload, X, Loader2, Layers, Gauge, Waypoints } from "lucide-react";
import * as SelectPrimitive from "@radix-ui/react-select";
import * as SliderPrimitive from "@radix-ui/react-slider";
import { analyzeImage, getModels, saveScanToFirestore, type AnalysisResult, type ModelInfo } from "../api/client";
import { useAuth } from "../contexts/AuthContext";
import FindingsTable from "../components/FindingsTable";

const ACCEPT = ".jpg,.jpeg,.png,.bmp,.tiff,.tif";

export default function AnalyzeScan() {
  const { user } = useAuth();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [conf, setConf] = useState(0.5);
  const [toothAssign, setToothAssign] = useState(false);
  const [patientName, setPatientName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [openPopover, setOpenPopover] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getModels()
      .then(setModels)
      .catch(() => {
        setError("Could not load models. Check backend URL/CORS settings.");
      });
  }, []);

  useEffect(() => {
    if (!models.length || selectedModel) return;

    const preferred = models.find((m) => {
      const raw = `${m.name} ${m.path}`.toLowerCase();
      return raw.includes("pano_gpu2") || raw.includes("pano_caries_only_gpu2");
    });

    setSelectedModel(preferred?.path || models[0].path);
  }, [models, selectedModel]);

  useEffect(() => {
    if (!file) { setPreview(null); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError("");
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const handleAnalyze = async () => {
    if (!file || !selectedModel) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await analyzeImage(
        file,
        selectedModel,
        conf,
        selectedModel.toLowerCase().includes("bitewing") ? "Bitewing" : "Panoramic",
        toothAssign,
        patientName,
      );
      setResult(res);
      // Save to Firestore
      if (user) {
        saveScanToFirestore(user.uid, {
          filename: res.filename,
          patientName,
          suspicion: res.suspicion_level,
          confidence: res.overall_confidence,
          detectionsCount: res.num_detections,
          modality: res.modality,
          turnaroundS: res.turnaround_s,
        }).catch(console.error);
      }
    } catch (e: any) {
      setError(e.message || "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const MODEL_DISPLAY: Record<string, string> = {
    "pano_gpu2": "Panoramic",
    "pano_caries_only_gpu2": "Panoramic",
    "bitewing": "Bitewing",
    "bitewing_caries_only": "Bitewing",
    "potato": "Kiwi",
    "pano_dc1000_potato": "Kiwi",
  };

  const modelOptions = useMemo(() => (
    models.map((m) => {
      const raw = m.name.toLowerCase();
      const path = m.path.toLowerCase();

      let label = m.name;
      for (const [key, value] of Object.entries(MODEL_DISPLAY)) {
        if (raw.includes(key) || path.includes(key)) {
          label = value;
          break;
        }
      }

      return { value: m.path, label };
    })
  ), [models]);

  const currentModelLabel = modelOptions.find((o) => o.value === selectedModel)?.label || "Panoramic";

  const toolbarBtnStyle = (active: boolean = false): React.CSSProperties => ({
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "6px 10px",
    borderRadius: 8,
    fontSize: 12,
    fontWeight: 500,
    border: "1px solid var(--border-color)",
    cursor: "pointer",
    transition: "all 0.15s",
    background: active ? "var(--color-leaf-subtle)" : "var(--color-surface)",
    color: active ? "var(--color-leaf-text)" : "var(--color-ink-secondary)",
    fontFamily: "var(--font-body)",
  });

  return (
    <div style={{
      flex: 1,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: result ? "flex-start" : "center",
      padding: "32px 32px 24px",
      maxWidth: 900,
      width: "100%",
      margin: "0 auto",
      minHeight: 0,
    }}>
      {/* Header row: title + patient name inline */}
      <div style={{
        display: "flex",
        alignItems: "baseline",
        gap: 24,
        width: "100%",
        marginBottom: 24,
      }}>
        <h1 style={{
          fontFamily: "var(--font-display)",
          fontSize: 26,
          fontWeight: 400,
          color: "var(--color-ink)",
          whiteSpace: "nowrap",
          margin: 0,
        }}>
          Analyze X-ray
        </h1>
        <input
          style={{
            flex: 1,
            padding: "8px 14px",
            background: "transparent",
            border: "1px solid var(--border-color)",
            borderRadius: 8,
            fontSize: 14,
            fontFamily: "var(--font-body)",
            color: "var(--color-ink)",
            outline: "none",
            transition: "border-color 0.15s",
          }}
          type="text"
          placeholder="Patient name (optional)"
          value={patientName}
          onChange={(e) => setPatientName(e.target.value)}
          onFocus={(e) => e.currentTarget.style.borderColor = "var(--color-leaf)"}
          onBlur={(e) => e.currentTarget.style.borderColor = "var(--border-color)"}
        />
      </div>

      {/* Upload / Image area */}
      <div style={{ width: "100%", marginBottom: 16 }}>
        {!file ? (
          <div
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            style={{
              width: "100%",
              border: `1.5px dashed ${dragOver ? "var(--color-leaf)" : "var(--border-emphasis)"}`,
              borderRadius: 14,
              padding: "56px 32px",
              textAlign: "center",
              cursor: "pointer",
              transition: "all 0.2s",
              background: dragOver ? "var(--color-leaf-subtle)" : "var(--color-surface)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Upload size={36} strokeWidth={1.5} style={{ color: "var(--color-ink-ghost)", marginBottom: 10 }} />
            <div style={{ fontSize: 15, color: "var(--color-ink-secondary)" }}>
              Drag an X-ray here, or click to select
            </div>
            <div style={{ fontSize: 12, color: "var(--color-ink-tertiary)", marginTop: 4 }}>
              JPG, PNG, BMP, TIFF
            </div>
          </div>
        ) : (
          <div style={{ position: "relative" }} className="group">
            <div style={{ width: "100%", background: "#1a1a1a", borderRadius: 14, overflow: "hidden" }}>
              <img
                src={result ? result.annotated_image_url : preview!}
                alt="X-ray"
                style={{ width: "100%", display: "block" }}
              />
            </div>
            {!loading && (
              <button
                onClick={clearFile}
                className="opacity-0 group-hover:opacity-100"
                style={{
                  position: "absolute", top: 10, right: 10, width: 30, height: 30,
                  borderRadius: "50%", background: "rgba(0,0,0,0.5)", color: "white",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  border: "none", cursor: "pointer", transition: "opacity 0.2s",
                }}
              >
                <X size={14} />
              </button>
            )}
            {!result && (
              <div style={{
                position: "absolute", bottom: 0, left: 0, right: 0, padding: "12px 16px",
                background: "linear-gradient(to top, rgba(0,0,0,0.55), transparent)",
                borderRadius: "0 0 14px 14px", display: "flex", alignItems: "center", justifyContent: "space-between",
              }}>
                <span style={{ color: "rgba(255,255,255,0.7)", fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginRight: 12 }}>
                  {file.name}
                </span>
                <button
                  onClick={handleAnalyze}
                  disabled={loading}
                  style={{
                    display: "flex", alignItems: "center", gap: 6, padding: "8px 18px",
                    borderRadius: 8, fontSize: 13, fontWeight: 500, border: "none",
                    cursor: loading ? "default" : "pointer", transition: "all 0.15s",
                    background: loading ? "rgba(255,255,255,0.2)" : "var(--color-leaf)",
                    color: loading ? "rgba(255,255,255,0.6)" : "white", flexShrink: 0,
                  }}
                >
                  {loading ? (<><Loader2 size={14} className="animate-spin" /> Analyzing...</>) : "Analyze"}
                </button>
              </div>
            )}
            {result && (
              <div style={{ position: "absolute", top: 10, left: 10 }}>
                <span className={`verdict-badge ${result.suspicion_level.toLowerCase()}`}>{result.suspicion_level}</span>
              </div>
            )}
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
            e.currentTarget.value = "";
          }}
        />
      </div>

      {/* Settings toolbar */}
      <div style={{ display: "flex", alignItems: "center", gap: 6, width: "100%", marginBottom: 20, flexWrap: "wrap" }}>

        {/* Model — direct Radix Select, no popover wrapper */}
        <SelectPrimitive.Root value={selectedModel} onValueChange={setSelectedModel}>
          <SelectPrimitive.Trigger style={toolbarBtnStyle()}>
            <Layers size={13} />
            <SelectPrimitive.Value>{currentModelLabel}</SelectPrimitive.Value>
          </SelectPrimitive.Trigger>
          <SelectPrimitive.Portal>
            <SelectPrimitive.Content
              style={{
                background: "var(--color-surface)",
                border: "1px solid var(--border-emphasis)",
                borderRadius: 10,
                boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
                overflow: "hidden",
                zIndex: 100,
                minWidth: 180,
              }}
              position="popper"
              sideOffset={6}
              side="bottom"
              align="start"
            >
              <SelectPrimitive.Viewport style={{ padding: 4 }}>
                {modelOptions.map(opt => (
                  <SelectPrimitive.Item
                    key={opt.value}
                    value={opt.value}
                    style={{
                      padding: "7px 12px",
                      fontSize: 13,
                      color: "var(--color-ink)",
                      borderRadius: 6,
                      cursor: "pointer",
                      outline: "none",
                      transition: "background 0.1s",
                    }}
                    className="data-[highlighted]:bg-leaf-subtle"
                  >
                    <SelectPrimitive.ItemText>{opt.label}</SelectPrimitive.ItemText>
                  </SelectPrimitive.Item>
                ))}
              </SelectPrimitive.Viewport>
            </SelectPrimitive.Content>
          </SelectPrimitive.Portal>
        </SelectPrimitive.Root>

        {/* Confidence — popover below */}
        <div style={{ position: "relative" }}>
          <button
            style={toolbarBtnStyle(openPopover === "conf")}
            onClick={() => setOpenPopover(openPopover === "conf" ? null : "conf")}
          >
            <Gauge size={13} />
            {Math.round(conf * 100)}%
          </button>
          {openPopover === "conf" && (
            <>
              <div style={{ position: "fixed", inset: 0, zIndex: 40 }} onClick={() => setOpenPopover(null)} />
              <div style={{
                position: "absolute",
                top: "calc(100% + 6px)",
                left: 0,
                background: "var(--color-surface)",
                border: "1px solid var(--border-emphasis)",
                borderRadius: 10,
                boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
                padding: 16,
                zIndex: 50,
                width: 220,
              }}>
                <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-ink-tertiary)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 10, fontFamily: "var(--font-display)" }}>
                  Confidence threshold
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <SliderPrimitive.Root
                    value={[conf]}
                    onValueChange={([v]) => setConf(v)}
                    min={0.05}
                    max={0.95}
                    step={0.05}
                    style={{ position: "relative", display: "flex", alignItems: "center", flex: 1, height: 20, userSelect: "none", touchAction: "none" }}
                  >
                    <SliderPrimitive.Track style={{ position: "relative", height: 3, width: "100%", borderRadius: 9999, background: "var(--color-surface-inset)", flexGrow: 1 }}>
                      <SliderPrimitive.Range style={{ position: "absolute", height: "100%", borderRadius: 9999, background: "var(--color-leaf)" }} />
                    </SliderPrimitive.Track>
                    <SliderPrimitive.Thumb style={{ display: "block", width: 16, height: 16, borderRadius: "50%", background: "white", border: "2px solid var(--color-leaf)", boxShadow: "0 1px 3px rgba(0,0,0,0.12)", cursor: "pointer", outline: "none" }} />
                  </SliderPrimitive.Root>
                  <span style={{ fontSize: 13, fontWeight: 500, color: "var(--color-ink)", fontFamily: "var(--font-body)", minWidth: 36, textAlign: "right" }}>
                    {Math.round(conf * 100)}%
                  </span>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Tooth assignment toggle */}
        <button
          onClick={() => setToothAssign(!toothAssign)}
          style={toolbarBtnStyle(toothAssign)}
          title="Tooth assignment"
        >
          <Waypoints size={13} />
          Tooth
        </button>

        {file && (
          <button
            onClick={handleAnalyze}
            disabled={loading}
            style={{
              ...toolbarBtnStyle(false),
              background: loading ? "var(--color-surface-inset)" : "var(--color-leaf)",
              color: loading ? "var(--color-ink-tertiary)" : "white",
              border: "none",
              fontWeight: 600,
            }}
          >
            {loading ? (
              <>
                <Loader2 size={13} className="animate-spin" />
                Analyzing...
              </>
            ) : result ? "Analyze again" : "Analyze"}
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div style={{ color: "var(--color-high)", fontWeight: 500, fontSize: 13, marginBottom: 12, width: "100%" }}>
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div style={{ width: "100%" }}>
          <FindingsTable detections={result.detections} />
        </div>
      )}
    </div>
  );
}
