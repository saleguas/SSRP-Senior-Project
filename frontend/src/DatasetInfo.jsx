export default function DatasetInfo() {
    return (
      <div className="card">
        <h2>🐟 Dataset Info</h2>
  
        <p>
          This project supports multiple fish tracking datasets used for training and evaluation.
        </p>
  
        <div className="block">
          <h3>Available Datasets</h3>
          <ul className="dataset-list">
            <li>
              🐠 <strong>aau-zebrafish-reid</strong>
              <p>Zebrafish dataset used for tracking and re-identification tasks.</p>
            </li>
            <li>
              🌊 <strong>deep-vision-fish</strong>
              <p>General dataset for fish detection across different environments.</p>
            </li>
            <li>
              📹 <strong>noaa-puget-sound-nearshore-fish</strong>
              <p>Real-world underwater footage collected from NOAA surveys.</p>
            </li>
          </ul>
        </div>
  
        <div className="block">
          <h3>How to Use</h3>
          <ul>
            <li>Use <strong>Run Tracking</strong> to process videos</li>
            <li>Use <strong>Visualize</strong> to generate output videos</li>
            <li>Use <strong>Dataset Info</strong> to query dataset details</li>
          </ul>
        </div>
  
        <div className="block">
          <h3>Tip</h3>
          <p>
            You can enter dataset names or aliases (like "zebrafish") in the CLI or UI.
          </p>
        </div>
      </div>
    )
  }