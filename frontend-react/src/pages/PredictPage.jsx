import { useState } from "react";
import PredictionForm from "../components/PredictionForm";
import PredictionResultCard from "../components/PredictionResultCard";
import { predictReturn } from "../api";

export default function PredictPage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (formData) => {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const data = await predictReturn(formData);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || "Prediction failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h2>Predict Return Risk</h2>
        <p className="page-desc">Enter order details to predict the likelihood of a product return.</p>
      </div>

      <PredictionForm onSubmit={handleSubmit} loading={loading} />

      {error && <div className="error-state">{error}</div>}

      <PredictionResultCard result={result} />
    </div>
  );
}
