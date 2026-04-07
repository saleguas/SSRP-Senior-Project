export default function DatasetInfo() {
    return (
      <div className="card">
        <h2>Dataset Info</h2>
  
        <p>
          This project supports multiple fish tracking datasets used for training and evaluation.
        </p>
  
        <ul>
          <li><strong>aau-zebrafish-reid</strong> – Training dataset for zebrafish tracking</li>
          <li><strong>deep-vision-fish</strong> – General fish detection dataset</li>
          <li><strong>noaa-puget-sound-nearshore-fish</strong> – Real-world underwater footage</li>
        </ul>
  
        <p>
          You can use the CLI or buttons in this UI to explore dataset details and run tracking.
        </p>
      </div>
    )
  }