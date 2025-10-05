"use client";

import React, { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { type MetricsFull } from "@/lib/api";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell,
} from "recharts";
import { formatPercent } from "@/lib/utils";

type Props = { metrics: MetricsFull | null; loading?: boolean };

export default function Insights({ metrics, loading = false }: Props) {
  if (loading) {
    return <Card><CardHeader><CardTitle>Flight Data</CardTitle></CardHeader><CardContent>Loading…</CardContent></Card>;
  }
  if (!metrics) {
    return <Card><CardHeader><CardTitle>Flight Data</CardTitle></CardHeader><CardContent>No metrics available.</CardContent></Card>;
  }

  // Try to derive class labels from metrics; fall back to common Kepler labels.
  const classLabels = useMemo(() => {
    const keys = Object.keys(metrics.auc_per_class ?? {});
    return keys.length ? keys : ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"];
  }, [metrics]);

  // Feature importance (prefer SHAP, else gain)
  const importance = useMemo(() => {
    const src = metrics.feature_importances_shap ?? metrics.feature_importances_gain ?? {};
    return Object.entries(src)
      .sort(([, a], [, b]) => (b as number) - (a as number))
      .slice(0, 10)
      .map(([feature, score]) => ({ feature, importance: (score as number) * 100 }));
  }, [metrics]);

  // ROC series (each class is its own series)
  const rocSeries = useMemo(() => {
    return Object.entries(metrics.roc || {}).map(([label, data]) => ({
      label,
      points: (data.fpr ?? []).map((fpr, i) => ({
        x: (fpr ?? 0) * 100,
        y: (data.tpr?.[i] ?? 0) * 100,
      })),
      auc: (metrics.auc_per_class || {})[label],
    }));
  }, [metrics]);

  // PR series
  const prSeries = useMemo(() => {
    return Object.entries(metrics.pr || {}).map(([label, data]) => ({
      label,
      points: (data.recall ?? []).map((rec, i) => ({
        x: (rec ?? 0) * 100,
        y: (data.precision?.[i] ?? 0) * 100,
      })),
      ap: data.ap,
    }));
  }, [metrics]);

  // Calibration curve
  const calibration = useMemo(() => {
    const bins = metrics.calibration_bins;
    if (!bins?.bin_mid?.length) return [];
    return bins.bin_mid.map((m, i) => ({
      expected: (m ?? 0) * 100,
      actual: (bins.acc?.[i] ?? 0) * 100,
      count: bins.count?.[i] ?? 0,
    }));
  }, [metrics]);

  // Confusion matrix (heat table)
  const cm = metrics.confusion_matrix || [];
  const cmMax = cm.flat().reduce((m, n) => Math.max(m, n), 0) || 1;

  const meanAuc = useMemo(() => {
    const vals = Object.values(metrics.auc_per_class || {});
    if (!vals.length) return 0;
    const s = vals.reduce((a, b) => a + b, 0);
    return s / vals.length;
  }, [metrics]);

  return (
    <div className="grid gap-6">
      {/* Summary */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="CV Macro F1" value={(metrics.search?.cv_macro_f1_mean ?? 0).toFixed(3)} />
        <StatCard label="ECE" value={(metrics.ece ?? 0).toFixed(3)} />
        <StatCard label="Optimal Threshold" value={(metrics.recommended_threshold ?? 0.5).toFixed(2)} />
        <StatCard label="Mean AUC" value={meanAuc.toFixed(3)} />
      </div>

      {/* Feature importance */}
      <Card>
        <CardHeader><CardTitle>Feature Importance (Top 10)</CardTitle></CardHeader>
        <CardContent className="h-72">
          <ResponsiveContainer>
            <BarChart data={importance} layout="vertical" margin={{ left: 12, right: 12, top: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" unit="%" />
              <YAxis dataKey="feature" type="category" width={140} />
              <Tooltip />
              <Bar dataKey="importance">
                {importance.map((_, i) => <Cell key={i} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* ROC curves */}
      <Card>
        <CardHeader><CardTitle>ROC Curves</CardTitle></CardHeader>
        <CardContent className="h-72">
          <ResponsiveContainer>
            <LineChart margin={{ left: 12, right: 12, top: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="FPR" unit="%" />
              <YAxis type="number" dataKey="y" name="TPR" unit="%" domain={[0, 100]} />
              <Tooltip />
              <Legend />
              {rocSeries.map((s) => (
                <Line
                  key={s.label}
                  data={s.points}
                  type="monotone"
                  dataKey="y"
                  dot={false}
                  name={`${s.label}${s.auc ? ` (AUC ${s.auc.toFixed(3)})` : ""}`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* PR curves */}
      <Card>
        <CardHeader><CardTitle>Precision-Recall Curves</CardTitle></CardHeader>
        <CardContent className="h-72">
          <ResponsiveContainer>
            <LineChart margin={{ left: 12, right: 12, top: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="Recall" unit="%" domain={[0, 100]} />
              <YAxis type="number" dataKey="y" name="Precision" unit="%" domain={[0, 100]} />
              <Tooltip />
              <Legend />
              {prSeries.map((s) => (
                <Line
                  key={s.label}
                  data={s.points}
                  type="monotone"
                  dataKey="y"
                  dot={false}
                  name={`${s.label}${s.ap != null ? ` (AP ${s.ap.toFixed(3)})` : ""}`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Calibration */}
      <Card>
        <CardHeader><CardTitle>Calibration</CardTitle></CardHeader>
        <CardContent className="h-64">
          <ResponsiveContainer>
            <LineChart data={calibration} margin={{ left: 12, right: 12, top: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="expected" unit="%" />
              <YAxis unit="%" domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="expected" name="Expected" dot={false} />
              <Line type="monotone" dataKey="actual" name="Actual" dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-2 text-xs text-slate-400">
            Reliability diagram built from calibration bins (count-weighted).
          </div>
        </CardContent>
      </Card>

      {/* Confusion matrix */}
      <Card>
        <CardHeader><CardTitle>Confusion Matrix</CardTitle></CardHeader>
        <CardContent>
          <div className="overflow-auto">
            <table className="min-w-[480px] border-collapse">
              <thead>
                <tr>
                  <th className="p-2 text-left text-xs text-slate-400">True \ Pred</th>
                  {classLabels.map((l) => (
                    <th key={l} className="p-2 text-xs text-slate-400">{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cm.map((row, rIdx) => (
                  <tr key={rIdx}>
                    <td className="p-2 text-xs text-slate-400">{classLabels[rIdx] ?? `C${rIdx}`}</td>
                    {row.map((val, cIdx) => {
                      const alpha = cmMax ? Math.min(1, val / cmMax) : 0;
                      return (
                        <td key={cIdx} className="p-2 text-center text-sm"
                            style={{ backgroundColor: `rgba(245, 158, 11, ${0.15 + 0.6 * alpha})` }}>
                          {val}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardHeader><CardTitle className="text-sm text-slate-400">{label}</CardTitle></CardHeader>
      <CardContent className="text-2xl font-semibold">{value}</CardContent>
    </Card>
  );
}
