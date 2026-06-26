import React, { useState, useEffect } from 'react';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip,
  ResponsiveContainer, AreaChart, Area, CartesianGrid
} from 'recharts';
import { fetchMetrics } from '../api/metrics';
import { useSession } from '../hooks/useSession';
import styles from './AnalyticsDashboard.module.css';

const CATEGORY_COLORS = {
  dsa: '#f59e0b',
  evaluate: '#10b981',
  math: '#3b82f6',
  general: '#64748b',
  manual: '#a855f7'
};

const CHART_COLORS = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444'];

export default function AnalyticsDashboard() {
  const sessionId = useSession();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const metrics = await fetchMetrics(sessionId);
        setData(metrics);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [sessionId]);

  if (loading) return <div className={styles.loading}>Loading analytics...</div>;
  if (error) return <div className={styles.error}>Failed to load analytics: {error}</div>;
  if (!data) return null;

  // Prepare Pie Chart Data
  const pieData = Object.entries(data.category_breakdown).map(([name, value]) => ({
    name, value
  }));

  // Prepare Bar Chart Data
  const barData = Object.entries(data.model_usage).map(([name, value]) => ({
    name, value
  }));

  return (
    <div className={styles.dashboardContainer}>
      <div className={styles.header}>
        <h1>Analytics Dashboard</h1>
        <p>Real-time insights into your session usage and routing patterns.</p>
      </div>

      <div className={styles.statsGrid}>
        <div className={styles.statCard}>
          <span className={styles.statLabel}>Total Queries</span>
          <span className={styles.statValue}>{data.total_queries}</span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statLabel}>Tokens Used</span>
          <span className={styles.statValue}>{data.total_tokens.toLocaleString()}</span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statLabel}>Tokens Saved</span>
          <span className={`${styles.statValue} ${styles.highlight}`}>
            {data.tokens_saved.toLocaleString()}
          </span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statLabel}>Avg Reduction</span>
          <span className={`${styles.statValue} ${styles.highlight}`}>
            {data.avg_reduction_pct}%
          </span>
        </div>
      </div>

      <div className={styles.chartsGrid}>
        {/* Category Breakdown Pie Chart */}
        <div className={styles.chartCard}>
          <h2>Intent Classification</h2>
          <div className={styles.chartWrapper}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  stroke="none"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {pieData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={CATEGORY_COLORS[entry.name] || CHART_COLORS[index % CHART_COLORS.length]} 
                    />
                  ))}
                </Pie>
                <RechartsTooltip 
                  contentStyle={{ backgroundColor: 'var(--bg-primary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
                  itemStyle={{ color: 'var(--text-primary)' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Model Usage Bar Chart */}
        <div className={styles.chartCard}>
          <h2>Model Usage</h2>
          <div className={styles.chartWrapper}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" horizontal={false} />
                <XAxis type="number" stroke="var(--text-secondary)" />
                <YAxis dataKey="name" type="category" stroke="var(--text-secondary)" width={100} tick={{ fontSize: 12 }} />
                <RechartsTooltip 
                  contentStyle={{ backgroundColor: 'var(--bg-primary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
                  cursor={{ fill: 'var(--bg-primary)' }}
                />
                <Bar dataKey="value" fill="var(--purple)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Savings Timeline Area Chart */}
        <div className={`${styles.chartCard} ${styles.fullWidth}`}>
          <h2>Token Savings Timeline</h2>
          <div className={styles.chartWrapper}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data.savings_timeline} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorSaved" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--purple)" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="var(--purple)" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                <XAxis dataKey="date" stroke="var(--text-secondary)" tick={{ fontSize: 12 }} />
                <YAxis stroke="var(--text-secondary)" tick={{ fontSize: 12 }} />
                <RechartsTooltip 
                  contentStyle={{ backgroundColor: 'var(--bg-primary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
                />
                <Area type="monotone" dataKey="saved" stroke="var(--purple)" fillOpacity={1} fill="url(#colorSaved)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
