export default function PredictionResultCard({ result }) {
  if (!result) return null;

  const { prediction_label, probability, decision_threshold, risk_level, recommendation } = result;
  const isReturn = prediction_label === "Yes";
  const probPct = probability != null ? (probability * 100).toFixed(1) : "—";
  const riskClass = `risk-${(risk_level || "low").toLowerCase()}`;

  return (
    <div className="result-card fade-in">
      <div className="result-header">
        <h3>Prediction Result</h3>
        <span className={`result-label ${isReturn ? "result-label-return" : "result-label-safe"}`}>
          {isReturn ? "Likely Return" : "Unlikely Return"}
        </span>
      </div>

      <div className="result-metrics">
        <div className="metric-box">
          <span className="metric-value">{probPct}%</span>
          <span className="metric-label">Return Probability</span>
          <div className="progress-bar">
            <div
              className={`progress-fill ${isReturn ? "progress-danger" : "progress-safe"}`}
              style={{ width: `${Math.min(Math.max((probability || 0) * 100, 0), 100)}%` }}
            />
          </div>
        </div>

        <div className="metric-box">
          <span className={`risk-badge ${riskClass}`}>
            {risk_level}
          </span>
          <span className="metric-label">Risk Level</span>
        </div>

        <div className="metric-box">
          <span className="metric-value">{decision_threshold != null ? decision_threshold.toFixed(2) : "—"}</span>
          <span className="metric-label">Threshold</span>
        </div>
      </div>

      {recommendation && (
        <div className="result-recommendation">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" /><path d="M12 16v-4" /><path d="M12 8h.01" />
          </svg>
          <p>{recommendation}</p>
        </div>
      )}
    </div>
  );
}
