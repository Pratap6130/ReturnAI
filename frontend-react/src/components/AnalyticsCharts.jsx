import { useState, useEffect, useMemo } from "react";
import {
  BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { getRecentPredictions } from "../api";
import useThemeStore from "../themeStore";

const COLORS = ["#22c55e", "#eab308", "#ef4444"];
const PIE_COLORS = ["#22c55e", "#ef4444"];

function getRiskLevel(probability) {
  if (probability >= 0.75) return "High";
  if (probability >= 0.5) return "Medium";
  return "Low";
}

export default function AnalyticsCharts() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterCategory, setFilterCategory] = useState("All");
  const theme = useThemeStore((s) => s.theme);
  const isDark = theme === "dark";
  const gridStroke = isDark ? "#2e2e45" : "#f0f0f0";
  const tooltipStyle = { borderRadius: 8, border: `1px solid ${isDark ? "#2e2e45" : "#e5e7eb"}`, fontSize: 13, background: isDark ? "#1a1a2e" : "#fff", color: isDark ? "#e4e4e7" : "#1a1a1a" };
  const tickColor = isDark ? "#a1a1aa" : undefined;

  useEffect(() => {
    getRecentPredictions(100)
      .then((rows) => {
        setData(rows);
        setError(null);
      })
      .catch((err) => {
        setError(err.response?.data?.detail || "Failed to load analytics data");
      })
      .finally(() => setLoading(false));
  }, []);

  const categories = useMemo(() => {
    const cats = [...new Set(data.map((r) => r.product_category))];
    return ["All", ...cats.sort()];
  }, [data]);

  const filtered = useMemo(() => {
    if (filterCategory === "All") return data;
    return data.filter((r) => r.product_category === filterCategory);
  }, [data, filterCategory]);

  const riskDistribution = useMemo(() => {
    const counts = { Low: 0, Medium: 0, High: 0 };
    filtered.forEach((r) => { counts[getRiskLevel(r.probability)]++; });
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [filtered]);

  const returnPie = useMemo(() => {
    const returned = filtered.filter((r) => r.prediction === 1).length;
    const notReturned = filtered.length - returned;
    return [
      { name: "No Return", value: notReturned },
      { name: "Return", value: returned },
    ];
  }, [filtered]);

  const categoryProb = useMemo(() => {
    const map = {};
    data.forEach((r) => {
      if (!map[r.product_category]) map[r.product_category] = { sum: 0, count: 0 };
      map[r.product_category].sum += r.probability;
      map[r.product_category].count++;
    });
    return Object.entries(map)
      .map(([name, v]) => ({ name, probability: +(v.sum / v.count * 100).toFixed(1) }))
      .sort((a, b) => b.probability - a.probability);
  }, [data]);

  if (loading) {
    return (
      <div className="loading-state">
        <span className="spinner" />
        <p>Loading analytics…</p>
      </div>
    );
  }

  if (error) return <div className="error-state">{error}</div>;

  if (data.length === 0) {
    return (
      <div className="empty-state">
        <p>No prediction data available for analytics. Make some predictions first!</p>
      </div>
    );
  }

  return (
    <div className="analytics-charts">
      <div className="analytics-filter">
        <label htmlFor="cat-filter">Filter by category:</label>
        <select
          id="cat-filter"
          value={filterCategory}
          onChange={(e) => setFilterCategory(e.target.value)}
        >
          {categories.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <span className="filter-count">{filtered.length} predictions</span>
      </div>

      <div className="charts-grid">
        <div className="chart-card">
          <h4>Risk Level Distribution</h4>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={riskDistribution}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis dataKey="name" tick={{ fontSize: 13, fill: tickColor }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 13, fill: tickColor }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                {riskDistribution.map((_, i) => (
                  <Cell key={i} fill={COLORS[i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h4>Return vs No Return</h4>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={returnPie}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={4}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {returnPie.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i]} />
                ))}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card chart-wide">
          <h4>Avg Return Probability by Category</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={categoryProb} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 13, fill: tickColor }} unit="%" />
              <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 12, fill: tickColor }} />
              <Tooltip
                formatter={(v) => `${v}%`}
                contentStyle={tooltipStyle}
              />
              <Legend />
              <Bar dataKey="probability" name="Avg Probability %" fill="#6366f1" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
