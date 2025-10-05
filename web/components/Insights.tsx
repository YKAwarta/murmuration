"use client";

import React from "react";
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
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend,
  Cell,
} from "recharts";

type Props = { metrics: MetricsFull | null; loading?: boolean };

export default function Insights({ metrics, loading = false }: Props) {
  if (loading) {
    return (
      <Card>
        <CardHeader><CardTitle>Insights (Flight Deck)</CardTitle></CardHeader>
        <CardContent>Loading…</CardContent>
      </Card>
    );
  }
  if (!metrics) {
    return (
      <Card>
        <CardHeader><CardTitle>Insights (Flight Deck)</CardTitle></CardHeader>
        <CardContent>No metrics available.</CardContent>
      </Card>
    );
  }

  // -------- helpers
  const num = (v: unknown, mul = 1) => {
    const n = Number(v);
    return Number.isFinite(n) ? n * mul : null;
  };
  const pct = (v: unknown) => num(v, 100);

  // -------- summary stats (safe)
  const cvMacro = Number(metrics?.search?.cv_macro_f1_mean ?? 0);
  const ece = Number(metrics?.ece ?? 0);
  const recThr = Number(metrics?.recommended_threshold ?? 0.5);

  // -------- importance (SHAP preferred, else gain)
  const rawImp = metrics.feature_importances_shap ?? metrics.feature_importances_gain ?? {};
  const importance = Object.entries(rawImp)
    .map(([feature, score]) => ({ feature, importance: pct(score) }))
    .filter((d) => d.importance !== null)
    .sort((a, b) => (b.importance as number) - (a.importance as number))
    .slice(0, 10) as { feature: string; importance: number }[];

  // -------- ROC curves
  const rocSeries = Object.entries(metrics.roc ?? {}).map(([label, curve]) => {
    const points =
      (curve?.fpr ?? []).map((f, i) => {
        const x = pct(f);
        const y = pct(curve?.tpr?.[i]);
        return x !== null && y !== null ? { x, y } : null;
      }).filter(Boolean) as { x: number; y: number }[];
    return { label, points, auc: metrics.auc_per_class?.[label] };
  });

  // -------- PR curves
  const prSeries = Object.entries(metrics.pr ?? {}).map(([label, curve]) => {
    const points =
      (curve?.recall ?? []).map((r, i) => {
        const x = pct(r);
        const y = pct(curve?.precision?.[i]);
        return x !== null && y !== null ? { x, y } : null;
      }).filter(Boolean) as { x: number; y: number }[];
    return { label, points, ap: curve?.ap };
  });

  // -------- Calibration
  const calBins = metrics.calibration_bins;
  const calibration =
    calBins?.bin_mid?.map((m, i) => {
      const expected = pct(m);
      const actual = pct(calBins.acc?.[i]);
      const count = Number(calBins.count?.[i] ?? 0);
      return expected !== null && actual !== null ? { expected, actual, count } : null;
    }).filter(Boolean) ?? [];

  // -------- Confusion matrix
  const cm = (metrics.confusion_matrix ?? []).map((row) =>
    row.map((v) => Number(v) || 0)
  );
  const cmMax = cm.flat().reduce((m, n) => Math.max(m, n), 0) || 1;
  const classLabels = Object.keys(metrics.auc_per_class ?? {}).length
    ? Object.keys(metrics.auc_per_class!)
    : ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"];

  // -------- Mean AUC (safe)
  const aucVals = Object.values(metrics.auc_per_class ?? {}).map((v) => Number(v) || 0);
  const meanAuc = aucVals.length ? aucVals.reduce((a, b) => a + b, 0) / aucVals.length : 0;

  return (
    <div className="grid gap-6">
      {/* Summary */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="CV Macro F1" value={cvMacro.toFixed(3)} />
        <StatCard label="ECE" value={ece.toFixed(3)} />
        <StatCard label="Optimal Threshold" value={recThr.toFixed(2)} />
        <StatCard label="Mean AUC" value={meanAuc.toFixed(3)} />
      </div>

      {/* Feature Importance (show only if provided by API) */}
{importance.length > 0 && (
  <Card>
    <CardHeader><CardTitle>Feature Importance (Top 10)</CardTitle></CardHeader>
    <CardContent className="h-72">
      <ResponsiveContainer>
        <BarChart data={importance} layout="vertical" margin={{ left: 12, right: 12, top: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" unit="%" domain={[0, "dataMax"]} />
          <YAxis dataKey="feature" type="category" width={140} />
          <RechartsTooltip formatter={(v: unknown) => `${(Number(v) || 0).toFixed(1)}%`} />
          <Bar dataKey="importance">
            {importance.map((_, i) => <Cell key={i} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
)}

      {/* ROC */}
      <Card>
        <CardHeader><CardTitle>ROC Curves</CardTitle></CardHeader>
        <CardContent className="h-72">
          <ResponsiveContainer>
            <LineChart margin={{ left: 12, right: 12, top: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="FPR" unit="%" domain={[0, 100]} />
              <YAxis type="number" dataKey="y" name="TPR" unit="%" domain={[0, 100]} />
              <RechartsTooltip />
              <Legend />
              {rocSeries.map((s) => (
                <Line
                  key={s.label}
                  data={s.points}
                  type="monotone"
                  dataKey="y"
                  name={`${s.label}${s.auc != null ? ` (AUC ${Number(s.auc).toFixed(3)})` : ""}`}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* PR */}
      <Card>
        <CardHeader><CardTitle>Precision–Recall Curves</CardTitle></CardHeader>
        <CardContent className="h-72">
          <ResponsiveContainer>
            <LineChart margin={{ left: 12, right: 12, top: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="Recall" unit="%" domain={[0, 100]} />
              <YAxis type="number" dataKey="y" name="Precision" unit="%" domain={[0, 100]} />
              <RechartsTooltip />
              <Legend />
              {prSeries.map((s) => (
                <Line
                  key={s.label}
                  data={s.points}
                  type="monotone"
                  dataKey="y"
                  name={`${s.label}${s.ap != null ? ` (AP ${Number(s.ap).toFixed(3)})` : ""}`}
                  dot={false}
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
          {calibration.length === 0 ? (
            <div className="text-sm text-slate-400">No calibration bins provided by the API.</div>
          ) : (
            <ResponsiveContainer>
              <LineChart data={calibration} margin={{ left: 12, right: 12, top: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="expected" unit="%" domain={[0, 100]} />
                <YAxis unit="%" domain={[0, 100]} />
                <RechartsTooltip />
                <Legend />
                <Line type="monotone" dataKey="expected" name="Expected" dot={false} />
                <Line type="monotone" dataKey="actual" name="Actual" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      {/* Confusion Matrix */}
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
                      const alpha = Math.min(1, val / cmMax);
                      return (
                        <td
                          key={cIdx}
                          className="p-2 text-center text-sm"
                          style={{ backgroundColor: `rgba(245, 158, 11, ${0.15 + 0.6 * alpha})` }}
                        >
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
