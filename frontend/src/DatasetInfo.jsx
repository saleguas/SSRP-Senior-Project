import { useState } from "react"

export default function DatasetInfo() {
  const [search, setSearch] = useState("")
  const [selected, setSelected] = useState(null)

  const datasets = [
    {
      name: "aau-zebrafish-reid",
      emoji: "🐠",
      description: "Zebrafish tracking and re-identification dataset.",
      type: "training",
      aliases: ["aau", "zebrafish"],
      source: "Academic",
      size: "Medium"
    },
    {
      name: "deep-vision-fish",
      emoji: "🌊",
      description: "General fish detection dataset.",
      type: "detection",
      aliases: [],
      source: "Mixed",
      size: "Large"
    },
    {
      name: "noaa-puget-sound-nearshore-fish",
      emoji: "📹",
      description: "Real-world underwater NOAA dataset.",
      type: "real-world",
      aliases: ["noaa"],
      source: "NOAA",
      size: "Large"
    }
  ]

  const filtered = datasets.filter((d) =>
    d.name.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="card">
      <h2>🐟 Dataset Info</h2>

      <p>
        Explore available datasets used in fish tracking and detection tasks.
      </p>

      {/* SEARCH BAR */}
      <input
        placeholder="Search dataset..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

        <p style={{ marginTop: "12px", color: "#a8b0bf" }}>
        Available datasets are listed below:
        </p>

      {/* DATASET LIST */}
      <div className="dataset-grid">
        {filtered.map((d) => (
          <div
            key={d.name}
            className="dataset-card"
            onClick={() => setSelected(d)}
          >
            <h3>{d.emoji} {d.name}</h3>
            <p>{d.description}</p>

            <div className="dataset-meta">
              <span>Type: {d.type}</span>
              <span>Size: {d.size}</span>
            </div>

            {d.aliases.length > 0 && (
              <div className="dataset-aliases">
                Aliases: {d.aliases.join(", ")}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* SELECTED DATASET DETAILS */}
      {selected && (
        <div className="dataset-detail">
          <h3>📊 Selected Dataset</h3>

          <p><strong>Name:</strong> {selected.name}</p>
          <p><strong>Type:</strong> {selected.type}</p>
          <p><strong>Source:</strong> {selected.source}</p>
          <p><strong>Size:</strong> {selected.size}</p>

          {selected.aliases.length > 0 && (
            <p><strong>Aliases:</strong> {selected.aliases.join(", ")}</p>
          )}
        </div>
      )}

      {/* USAGE SECTION */}
      <div className="block">
        <h3>How to Use</h3>
        <ul>
          <li>Use Run Tracking to process videos</li>
          <li>Use Visualize to generate outputs</li>
          <li>Use CLI commands for advanced control</li>
        </ul>
      </div>

      {/* TIP */}
      <div className="block">
        <h3>Tip</h3>
        <p>
          You can use dataset aliases instead of full names when running commands.
        </p>
      </div>
    </div>
  )
}