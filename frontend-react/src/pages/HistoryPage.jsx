import HistoryTable from "../components/HistoryTable";

export default function HistoryPage() {
  return (
    <div className="page-container">
      <div className="page-header">
        <h2>Prediction History</h2>
        <p className="page-desc">Your most recent predictions at a glance.</p>
      </div>
      <HistoryTable />
    </div>
  );
}
