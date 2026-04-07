import { useState, useRef } from "react"

const API_BASE = "http://localhost:8000"

// ── helpers ──────────────────────────────────────────────────────────────────

async function post(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || "Request failed")
  return data
}

/** Pull checkpoint paths out of fish_cli.py stdout */
function parseCheckpoints(stdout = "") {
  const best = stdout.match(/Saved best weights:\s*(.+)/)?.[1]?.trim() ?? null
  const last = stdout.match(/Saved last weights:\s*(.+)/)?.[1]?.trim() ?? null
  const log  = stdout.match(/Training log:\s*(.+)/)?.[1]?.trim()          ?? null
  return { best, last, log }
}

function formatDuration(ms) {
  if (ms < 1000) return `${ms}ms`
  const s = Math.floor(ms / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  return `${m}m ${s % 60}s`
}

function timestamp() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
}

// ── sub-components ────────────────────────────────────────────────────────────

function StatusBadge({ success, running }) {
  if (running) return <span style={badge("running")}>● Running</span>
  if (success === null) return null
  return success
    ? <span style={badge("ok")}>✓ Success</span>
    : <span style={badge("err")}>✗ Failed</span>
}

function badge(kind) {
  const map = {
    running: { background: "#1a2e50", color: "#4cc9f0", border: "1px solid #2f6fed" },
    ok:      { background: "#0d2818", color: "#4ade80", border: "1px solid #16a34a" },
    err:     { background: "#2a0e0e", color: "#f87171", border: "1px solid #dc2626" },
  }
  return {
    ...map[kind],
    padding: "3px 10px",
    borderRadius: "999px",
    fontSize: "12px",
    fontWeight: 600,
    letterSpacing: "0.02em",
    whiteSpace: "nowrap",
  }
}

function CheckpointCard({ label, path, icon }) {
  const [copied, setCopied] = useState(false)

  function copy() {
    navigator.clipboard.writeText(path).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    })
  }

  return (
    <div style={{
      background: "#0f1115",
      border: "1px solid #2a3b55",
      borderRadius: "10px",
      padding: "12px 14px",
      display: "flex",
      gap: "12px",
      alignItems: "flex-start",
    }}>
      <span style={{ fontSize: "18px", lineHeight: 1, paddingTop: "2px" }}>{icon}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          fontSize: "11px",
          fontWeight: 700,
          color: "#6b7a94",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          marginBottom: "4px",
        }}>
          {label}
        </div>
        <div style={{
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          fontSize: "12px",
          color: "#c9d6ff",
          wordBreak: "break-all",
          lineHeight: 1.5,
        }}>
          {path}
        </div>
      </div>
      <button
        onClick={copy}
        title="Copy path"
        style={{
          background: copied
            ? "#0d2818"
            : "rgba(27, 34, 48, 0.7)",
          border: copied ? "1px solid #16a34a" : "1px solid #2a3b55",
          color: copied ? "#4ade80" : "#a8b0bf",
          padding: "5px 10px",
          borderRadius: "7px",
          fontSize: "11px",
          flexShrink: 0,
          transition: "all 0.2s",
          // override global button gradient for this small inline button
          backgroundImage: "none",
          boxShadow: "none",
          transform: "none",
        }}
      >
        {copied ? "✓ Copied" : "Copy"}
      </button>
    </div>
  )
}

function SessionCard({ session, index, total }) {
  const [expanded, setExpanded] = useState(index === 0)
  const { best, last, log } = parseCheckpoints(session.stdout)
  const hasCheckpoints = best || last || log

  return (
    <div style={{
      background: "rgba(22, 26, 34, 0.7)",
      border: `1px solid ${session.success === false ? "#3a1a1a" : "#2a3b55"}`,
      borderRadius: "14px",
      overflow: "hidden",
      marginBottom: "12px",
      backdropFilter: "blur(8px)",
      transition: "transform 0.2s ease",
    }}>
      {/* header row */}
      <div
        onClick={() => setExpanded(e => !e)}
        style={{
          padding: "14px 16px",
          display: "flex",
          alignItems: "center",
          gap: "12px",
          cursor: "pointer",
          userSelect: "none",
        }}
      >
        <span style={{ color: "#6b7a94", fontSize: "12px", fontVariantNumeric: "tabular-nums", minWidth: "28px" }}>
          #{total - index}
        </span>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: "13px",
            fontWeight: 600,
            color: "#e8ecf1",
            marginBottom: "2px",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}>
            {session.data || "(no data path)"}
          </div>
          <div style={{ fontSize: "11px", color: "#6b7a94" }}>
            {session.startedAt}
            {session.duration != null && ` · ${formatDuration(session.duration)}`}
            {session.model && ` · ${session.model}`}
          </div>
        </div>

        <StatusBadge success={session.success} running={session.running} />

        <span style={{
          color: "#6b7a94",
          fontSize: "12px",
          transform: expanded ? "rotate(180deg)" : "none",
          transition: "transform 0.2s",
        }}>▾</span>
      </div>

      {/* expanded body */}
      {expanded && (
        <div style={{ padding: "0 16px 16px" }}>

          {/* checkpoint summary */}
          {hasCheckpoints && (
            <div style={{ marginBottom: "14px" }}>
              <div style={{
                fontSize: "11px",
                fontWeight: 700,
                color: "#6b7a94",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                marginBottom: "8px",
              }}>
                Checkpoints Saved
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {best && <CheckpointCard label="Best Weights" path={best} icon="🏆" />}
                {last && <CheckpointCard label="Last Weights" path={last} icon="💾" />}
                {log  && <CheckpointCard label="Training Log"  path={log}  icon="📋" />}
              </div>
            </div>
          )}

          {!hasCheckpoints && session.success === true && (
            <div style={{ color: "#6b7a94", fontSize: "13px", fontStyle: "italic", marginBottom: "14px" }}>
              No checkpoint paths detected in output.
            </div>
          )}

          {/* stdout / stderr */}
          {(session.stdout || session.stderr) && (
            <div>
              <div style={{
                fontSize: "11px",
                fontWeight: 700,
                color: "#6b7a94",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                marginBottom: "8px",
              }}>
                Output
              </div>
              {session.stdout && (
                <pre style={{
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  background: "#0f1115",
                  border: "1px solid #242b38",
                  borderRadius: "10px",
                  padding: "12px",
                  fontSize: "12px",
                  color: "#c9d6ff",
                  maxHeight: "220px",
                  overflowY: "auto",
                  margin: "0 0 8px",
                  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                }}>
                  {session.stdout}
                </pre>
              )}
              {session.stderr && (
                <pre style={{
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  background: "#1a0e0e",
                  border: "1px solid #3a1a1a",
                  borderRadius: "10px",
                  padding: "12px",
                  fontSize: "12px",
                  color: "#f87171",
                  maxHeight: "140px",
                  overflowY: "auto",
                  margin: 0,
                  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                }}>
                  {session.stderr}
                </pre>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── main component ────────────────────────────────────────────────────────────

export default function TrainingSession() {
  const [form, setForm] = useState({
    data: "",
    output: "",
    model: "yolov8n.pt",
    epochs: "",
  })
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(false)
  const startTimeRef = useRef(null)

  function update(key, val) {
    setForm(f => ({ ...f, [key]: val }))
  }

  async function handleTrain() {
    if (!form.data) return
    setLoading(true)
    startTimeRef.current = Date.now()

    const pending = {
      id: crypto.randomUUID(),
      data: form.data,
      output: form.output,
      model: form.model,
      epochs: form.epochs,
      startedAt: timestamp(),
      running: true,
      success: null,
      stdout: "",
      stderr: "",
      duration: null,
    }
    setSessions(s => [pending, ...s])

    try {
      const body = {
        data: form.data,
        output: form.output || "",
        model: form.model,
        epochs: form.epochs ? Number(form.epochs) : 0,
      }

      let result
      try {
        result = await post("/train", body)
      } catch (e) {
        result = {
          success: false,
          returncode: -1,
          stdout: "",
          stderr: `API error: ${e.message}\n\nMake sure fish_api.py exposes a /train endpoint.`,
        }
      }

      const duration = Date.now() - startTimeRef.current
      setSessions(s =>
        s.map(sess =>
          sess.id === pending.id
            ? { ...sess, ...result, running: false, duration }
            : sess
        )
      )
    } finally {
      setLoading(false)
    }
  }

  const inputStyle = {
    width: "100%",
    padding: "10px 12px",
    borderRadius: "10px",
    border: "1px solid #2a2f3a",
    background: "#161a22",
    color: "#e8ecf1",
    fontSize: "14px",
    fontFamily: "inherit",
    outline: "none",
    boxSizing: "border-box",
  }

  return (
    <section className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "18px" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "18px", fontWeight: 700 }}>Training Session</h2>
          <p style={{ margin: "4px 0 0", color: "#6b7a94", fontSize: "13px" }}>
            Run training and track saved checkpoints
          </p>
        </div>
        {sessions.length > 0 && (
          <button
            onClick={() => setSessions([])}
            style={{
              background: "rgba(27, 34, 48, 0.7)",
              backgroundImage: "none",
              border: "1px solid #2a3b55",
              color: "#a8b0bf",
              fontSize: "12px",
              padding: "6px 12px",
              boxShadow: "none",
            }}
          >
            Clear history
          </button>
        )}
      </div>

      {/* form */}
      <div style={{
        background: "#0c111b",
        border: "1px solid #2a3b55",
        borderRadius: "12px",
        padding: "16px",
        marginBottom: "20px",
      }}>
        <label className="field">
          <span>Data path <span style={{ color: "#f87171" }}>*</span></span>
          <input
            style={inputStyle}
            value={form.data}
            onChange={e => update("data", e.target.value)}
            placeholder="Dataset folder, YOLO root, or JSON manifest"
          />
        </label>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
          <label className="field">
            <span>Output weights (.pt)</span>
            <input
              style={inputStyle}
              value={form.output}
              onChange={e => update("output", e.target.value)}
              placeholder="/path/to/output.pt"
            />
          </label>
          <label className="field">
            <span>Base model</span>
            <input
              style={inputStyle}
              value={form.model}
              onChange={e => update("model", e.target.value)}
              placeholder="yolov8n.pt"
            />
          </label>
        </div>

        <label className="field">
          <span>Epochs (0 = auto)</span>
          <input
            style={{ ...inputStyle, maxWidth: "180px" }}
            type="number"
            value={form.epochs}
            onChange={e => update("epochs", e.target.value)}
            placeholder="0"
          />
        </label>

        <button
          onClick={handleTrain}
          disabled={loading || !form.data}
          style={{ width: "100%", padding: "11px", fontWeight: 600, fontSize: "14px" }}
        >
          {loading ? "⏳ Training in progress…" : "▶ Start Training"}
        </button>
      </div>

      {/* session history */}
      {sessions.length > 0 && (
        <div>
          <div style={{
            fontSize: "11px",
            fontWeight: 700,
            color: "#6b7a94",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            marginBottom: "12px",
          }}>
            Session History — {sessions.length} run{sessions.length !== 1 ? "s" : ""}
          </div>
          {sessions.map((sess, i) => (
            <SessionCard key={sess.id} session={sess} index={i} total={sessions.length} />
          ))}
        </div>
      )}

      {sessions.length === 0 && (
        <div style={{
          border: "1px dashed #2a3b55",
          borderRadius: "12px",
          padding: "32px",
          textAlign: "center",
          color: "#6b7a94",
          fontSize: "13px",
        }}>
          No training sessions yet. Fill in a data path above and hit Start.
        </div>
      )}
    </section>
  )
}
