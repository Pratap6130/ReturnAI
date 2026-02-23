import { useState, useEffect } from "react";
import { getRecentPredictions } from "../api";



function getRiskLevel(probability) {
  if (probability >= 0.75) return "High";
  if (probability >= 0.5) return "Medium";
  return "Low";
}

export default function HistoryTable() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    getRecentPredictions(10)
      .then((data) => {
        setPredictions(data);
        setError(null);
      })
      .catch((err) => {
        setError(err.response?.data?.detail || "Failed to load predictions");
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="loading-state">
        <span className="spinner" />
        <p>Loading recent predictions…</p>
      </div>
    );
  }

  if (error) {
    return <div className="error-state">{error}</div>;
  }

  if (predictions.length === 0) {
    return (
      <div className="empty-state">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="16" y1="13" x2="8" y2="13" />
          <line x1="16" y1="17" x2="8" y2="17" />
          <polyline points="10 9 9 9 8 9" />
        </svg>
        <p>No predictions yet. Make your first prediction!</p>
      </div>
    );
  }

  return (
    <div className="table-wrapper">
      <table className="history-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Category</th>
            <th>Price</th>
            <th>Qty</th>
            <th>Age</th>
            <th>Gender</th>
            <th>Payment</th>
            <th>Shipping</th>
            <th>Probability</th>
            <th>Risk</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((row, i) => {
            const risk = getRiskLevel(row.probability);
            return (
              <tr key={row.id || i}>
                <td>{i + 1}</td>
                <td>{row.product_category}</td>
                <td>${Number(row.product_price).toFixed(2)}</td>
                <td>{row.order_quantity}</td>
                <td>{row.user_age}</td>
                <td>{row.user_gender}</td>
                <td>{row.payment_method}</td>
                <td>{row.shipping_method}</td>
                <td>{(row.probability * 100).toFixed(1)}%</td>
                <td>
                  <span className={`risk-badge-sm risk-${risk.toLowerCase()}`}>
                    {risk}
                  </span>
                </td>
                <td>
                  <span className={row.prediction === 1 ? "text-red" : "text-green"}>
                    {row.prediction === 1 ? "Return" : "No Return"}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
