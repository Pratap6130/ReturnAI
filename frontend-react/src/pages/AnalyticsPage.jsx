import AnalyticsCharts from "../components/AnalyticsCharts";

export default function AnalyticsPage() {
  return (
    <div className="page-container">
      <div className="page-header">
        <h2>Analytics</h2>
        <p className="page-desc">Visual insights from your prediction history.</p>
      </div>
      <AnalyticsCharts />
    </div>
  );
}
