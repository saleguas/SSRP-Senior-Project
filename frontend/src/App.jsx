import { useEffect, useMemo, useState } from "react"
import DatasetInfo from "./DatasetInfo"
const API_BASE = "http://localhost:8000"

const initialForms = {
  run: {
    data: "",
    output: "",
    weights: ""
  },
  visualize: {
    data: "",
    output: "",
    weights: "",
    coords_xlsx: "",
    fps: "30",
    duration_sec: "0"
  },
  visualizeBatch: {
    data: "",
    output: "",
    weights: "",
    write_tracks: false,
    fps: "30",
    duration_sec: "0"
  },
  validate: {
    data: "",
    output: "",
    weights: ""
  },
  modelInfo: {
    model: "yolov8n.pt",
    classes: "1"
  },
  checkPath: {
    path: ""
  },
  datasetInfo: {
    name: ""
  }
}

async function post(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  })

  const data = await res.json()

  if (!res.ok) {
    throw new Error(data.detail || "Request failed")
  }

  return data
}

async function get(path) {
  const res = await fetch(`${API_BASE}${path}`)
  const data = await res.json()

  if (!res.ok) {
    throw new Error(data.detail || "Request failed")
  }

  return data
}

function Section({ title, children }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      {children}
    </section>
  )
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  )
}

function Output({ result }) {
  if (!result) return null

  return (
    <div className="output">
      <div><strong>Success:</strong> {String(result.success)}</div>
      <div><strong>Exit Code:</strong> {result.returncode}</div>
      <div className="block">
        <strong>Stdout</strong>
        <pre>{result.stdout || "(empty)"}</pre>
      </div>
      <div className="block">
        <strong>Stderr</strong>
        <pre>{result.stderr || "(empty)"}</pre>
      </div>
    </div>
  )
}

export default function App() {
  const [forms, setForms] = useState(initialForms)
  const [activeTab, setActiveTab] = useState("run")
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [datasetsResult, setDatasetsResult] = useState(null)
  const [apiStatus, setApiStatus] = useState("Checking...")

  useEffect(() => {
    get("/health")
      .then(() => setApiStatus("Online"))
      .catch(() => setApiStatus("Offline"))
  }, [])

  const tabs = useMemo(
    () => [
      ["run", "Run Tracking"],
      ["visualize", "Visualize"],
      ["visualizeBatch", "Visualize Batch"],
      ["validate", "Validate"],
      ["modelInfo", "Model Info"],
      ["checkPath", "Check Path"],
      ["datasetInfo", "Dataset Info"]
    ],
    []
  )

  function updateForm(section, key, value) {
    setForms((prev) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }))
  }

  async function handleAction(action) {
    setLoading(true)
    setResult(null)

    try {
      if (action === "run") {
        setResult(await post("/run", forms.run))
      } else if (action === "visualize") {
        setResult(
          await post("/visualize", {
            ...forms.visualize,
            fps: Number(forms.visualize.fps),
            duration_sec: Number(forms.visualize.duration_sec)
          })
        )
      } else if (action === "visualizeBatch") {
        setResult(
          await post("/visualize-batch", {
            ...forms.visualizeBatch,
            fps: Number(forms.visualizeBatch.fps),
            duration_sec: Number(forms.visualizeBatch.duration_sec)
          })
        )
      } else if (action === "validate") {
        setResult(await post("/validate", forms.validate))
      } else if (action === "modelInfo") {
        setResult(
          await post("/model-info", {
            ...forms.modelInfo,
            classes: Number(forms.modelInfo.classes)
          })
        )
      } else if (action === "checkPath") {
        setResult(await post("/check-path", forms.checkPath))
      } else if (action === "datasetInfo") {
        setResult(await post("/dataset-info", forms.datasetInfo))
      }
    } catch (e) {
      setResult({
        success: false,
        returncode: -1,
        stdout: "",
        stderr: e.message
      })
    } finally {
      setLoading(false)
    }
  }

  async function handleListDatasets() {
    setLoading(true)
    setDatasetsResult(null)

    try {
      setDatasetsResult(await get("/datasets"))
    } catch (e) {
      setDatasetsResult({
        success: false,
        returncode: -1,
        stdout: "",
        stderr: e.message
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <div>
        <h1>🐟 Fish Tracking UI</h1>
        </div>
        <div className="status">API: {apiStatus}</div>
      </header>

      <div className="toolbar">
        <button onClick={handleListDatasets} disabled={loading}>
          List Datasets
        </button>
      </div>

      <Output result={datasetsResult} />

      <div className="tabs">
        {tabs.map(([key, label]) => (
          <button
            key={key}
            className={activeTab === key ? "tab active" : "tab"}
            onClick={() => setActiveTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {activeTab === "run" && (
        <Section title="Run Tracking">
          <Field label="Data path">
            <input
              value={forms.run.data}
              onChange={(e) => updateForm("run", "data", e.target.value)}
              placeholder="/absolute/path/to/video/or/frames"
            />
          </Field>
          <Field label="Output CSV">
            <input
              value={forms.run.output}
              onChange={(e) => updateForm("run", "output", e.target.value)}
              placeholder="/absolute/path/to/output.csv"
            />
          </Field>
          <Field label="Weights path">
            <input
              value={forms.run.weights}
              onChange={(e) => updateForm("run", "weights", e.target.value)}
              placeholder="/absolute/path/to/weights.pt"
            />
          </Field>
          <button onClick={() => handleAction("run")} disabled={loading}>
            {loading ? "Running..." : "Run"}
          </button>
        </Section>
      )}

      {activeTab === "visualize" && (
        <Section title="Visualize">
          <Field label="Data path">
            <input
              value={forms.visualize.data}
              onChange={(e) => updateForm("visualize", "data", e.target.value)}
              placeholder="/absolute/path/to/video/or/frames"
            />
          </Field>
          <Field label="Output MP4">
            <input
              value={forms.visualize.output}
              onChange={(e) => updateForm("visualize", "output", e.target.value)}
              placeholder="/absolute/path/to/output.mp4"
            />
          </Field>
          <Field label="Weights path">
            <input
              value={forms.visualize.weights}
              onChange={(e) => updateForm("visualize", "weights", e.target.value)}
              placeholder="/absolute/path/to/weights.pt"
            />
          </Field>
          <Field label="Coords XLSX">
            <input
              value={forms.visualize.coords_xlsx}
              onChange={(e) => updateForm("visualize", "coords_xlsx", e.target.value)}
              placeholder="/absolute/path/to/Fish coords.xlsx"
            />
          </Field>
          <Field label="FPS">
            <input
              value={forms.visualize.fps}
              onChange={(e) => updateForm("visualize", "fps", e.target.value)}
            />
          </Field>
          <Field label="Duration sec">
            <input
              value={forms.visualize.duration_sec}
              onChange={(e) => updateForm("visualize", "duration_sec", e.target.value)}
            />
          </Field>
          <button onClick={() => handleAction("visualize")} disabled={loading}>
            {loading ? "Running..." : "Visualize"}
          </button>
        </Section>
      )}

      {activeTab === "visualizeBatch" && (
        <Section title="Visualize Batch">
          <Field label="Data path">
            <input
              value={forms.visualizeBatch.data}
              onChange={(e) => updateForm("visualizeBatch", "data", e.target.value)}
              placeholder="/absolute/path/to/batch/folder"
            />
          </Field>
          <Field label="Output directory">
            <input
              value={forms.visualizeBatch.output}
              onChange={(e) => updateForm("visualizeBatch", "output", e.target.value)}
              placeholder="/absolute/path/to/output/dir"
            />
          </Field>
          <Field label="Weights path">
            <input
              value={forms.visualizeBatch.weights}
              onChange={(e) => updateForm("visualizeBatch", "weights", e.target.value)}
              placeholder="/absolute/path/to/weights.pt"
            />
          </Field>
          <Field label="FPS">
            <input
              value={forms.visualizeBatch.fps}
              onChange={(e) => updateForm("visualizeBatch", "fps", e.target.value)}
            />
          </Field>
          <Field label="Duration sec">
            <input
              value={forms.visualizeBatch.duration_sec}
              onChange={(e) =>
                updateForm("visualizeBatch", "duration_sec", e.target.value)
              }
            />
          </Field>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={forms.visualizeBatch.write_tracks}
              onChange={(e) =>
                updateForm("visualizeBatch", "write_tracks", e.target.checked)
              }
            />
            <span>Write tracks</span>
          </label>
          <button onClick={() => handleAction("visualizeBatch")} disabled={loading}>
            {loading ? "Running..." : "Visualize Batch"}
          </button>
        </Section>
      )}

      {activeTab === "validate" && (
        <Section title="Validate">
          <Field label="Data path">
            <input
              value={forms.validate.data}
              onChange={(e) => updateForm("validate", "data", e.target.value)}
              placeholder="/absolute/path/to/dataset/or/manifest"
            />
          </Field>
          <Field label="Output JSON">
            <input
              value={forms.validate.output}
              onChange={(e) => updateForm("validate", "output", e.target.value)}
              placeholder="/absolute/path/to/metrics.json"
            />
          </Field>
          <Field label="Weights path">
            <input
              value={forms.validate.weights}
              onChange={(e) => updateForm("validate", "weights", e.target.value)}
              placeholder="/absolute/path/to/weights.pt"
            />
          </Field>
          <button onClick={() => handleAction("validate")} disabled={loading}>
            {loading ? "Running..." : "Validate"}
          </button>
        </Section>
      )}

      {activeTab === "modelInfo" && (
        <Section title="Model Info">
          <Field label="Model">
            <input
              value={forms.modelInfo.model}
              onChange={(e) => updateForm("modelInfo", "model", e.target.value)}
            />
          </Field>
          <Field label="Classes">
            <input
              value={forms.modelInfo.classes}
              onChange={(e) => updateForm("modelInfo", "classes", e.target.value)}
            />
          </Field>
          <button onClick={() => handleAction("modelInfo")} disabled={loading}>
            {loading ? "Running..." : "Get Model Info"}
          </button>
        </Section>
      )}

      {activeTab === "checkPath" && (
        <Section title="Check Path">
          <Field label="Path">
            <input
              value={forms.checkPath.path}
              onChange={(e) => updateForm("checkPath", "path", e.target.value)}
              placeholder="/absolute/path/to/check"
            />
          </Field>
          <button onClick={() => handleAction("checkPath")} disabled={loading}>
            {loading ? "Running..." : "Check"}
          </button>
        </Section>
      )}

      {activeTab === "datasetInfo" && <DatasetInfo />}

      <Output result={result} />
    </div>
  )
}