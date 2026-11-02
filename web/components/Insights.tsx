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
} from "recharts";

type Props = { metrics: MetricsFull | null; loading?: boolean };

const SERIES_COLORS = ["#f59e0b", "#38bdf8", "#a78bfa"];

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

  const { test, calibration, decision } = metrics;
  const before = calibration.test_before;
  const after = calibration.test_after;
  const labels = metrics.labels;

  // -------- importance: share of total mean |TreeSHAP| (raw logit units)
  const shap = metrics.feature_importance.mean_abs_shap;
  const shapTotal = Object.values(shap).reduce((a, b) => a + b, 0) || 1;
  const importance = Object.entries(shap)
    .map(([feature, score]) => ({ feature, share: (score / shapTotal) * 100 }))
    .sort((a, b) => b.share - a.share)
    .slice(0, 10);

  // -------- ROC / PR curves (calibrated probabilities)
  const rocSeries = labels.map((label) => ({
    label,
    auc: test.auc_per_class[label],
    points: (test.roc[label]?.fpr ?? []).map((f, i) => ({ x: f * 100, y: (test.roc[label].tpr[i] ?? 0) * 100 })),
  }));
  const prSeries = labels.map((label) => ({
    label,
    ap: test.pr[label]?.ap,
    points: (test.pr[label]?.recall ?? []).map((r, i) => ({ x: r * 100, y: (test.pr[label].precision[i] ?? 0) * 100 })),
  }));

  // -------- reliability: mean confidence vs accuracy per bin, before and after
  const toPoints = (b: typeof before.reliability_bins) =>
    b.conf.map((c, i) => ({ x: c * 100, y: b.acc[i] * 100, n: b.count[i] }));
  const reliability = [
    { name: `Uncalibrated (ECE ${before.ece_top_label.toFixed(3)})`, color: "#ef4444", points: toPoints(before.reliability_bins) },
    { name: `Temperature-scaled (ECE ${after.ece_top_label.toFixed(3)})`, color: "#38bdf8", points: toPoints(after.reliability_bins) },
  ];

  // -------- confusion matrix
  const cm = test.confusion_matrix;
  const cmMax = cm.flat().reduce((m, n) => Math.max(m, n), 0) || 1;

  return (
    <div className="grid gap-6">
      {/* Summary (held-out test split) */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Test macro-F1" value={test.macro_f1.toFixed(3)} sub={`accuracy ${test.accuracy.toFixed(3)}`} />
        <StatCard label="ECE (top label)" value={`${before.ece_top_label.toFixed(3)} → ${after.ece_top_label.toFixed(3)}`} sub="before → after calibration" />
        <StatCard label="Brier score" value={`${before.brier.toFixed(3)} → ${after.brier.toFixed(3)}`} sub={`temperature T = ${calibration.temperature.toFixed(2)}`} />
        <StatCard
          label="Default threshold"
          value={decision.recommended_threshold.toFixed(2)}
          sub={`test: ${(decision.test.coverage * 100).toFixed(0)}% accepted, ${((decision.test.precision_on_accepted ?? 0) * 100).toFixed(1)}% precision`}
        />
      </div>

      {/* Calibration */}
      <Card>
        <CardHeader><CardTitle>Reliability Diagram (held-out test split)</CardTitle></CardHeader>
        <CardContent className="h-72">
          <ResponsiveContainer>
            <LineChart margin={{ left: 12, right: 12, top: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="Confidence" unit="%" domain={[30, 100]} />
              <YAxis type="number" dataKey="y" name="Accuracy" unit="%" domain={[30, 100]} />
              <RechartsTooltip formatter={(v: unknown) => `${(Number(v) || 0).toFixed(1)}%`} />
              <Legend />
              <Line
                data={[{ x: 30, y: 30 }, { x: 100, y: 100 }]}
                dataKey="y"
                name="Perfect calibration"
                stroke="#64748b"
                strokeDasharray="4 4"
                dot={false}
              />
              {reliability.map((s) => (
                <Line key={s.name} data={s.points} dataKey="y" name={s.name} stroke={s.color} dot />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Feature Importance */}
      {importance.length > 0 && (
        <Card>
          <CardHeader><CardTitle>Feature Importance (share of mean |SHAP|, top 10)</CardTitle></CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer>
              <BarChart data={importance} layout="vertical" margin={{ left: 12, right: 12, top: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  domain={[0, (max: number) => Math.ceil(max / 2) * 2]}
                  tickFormatter={(v: number) => `${v}%`}
                />
                <YAxis dataKey="feature" type="category" width={90} interval={0} />
                <RechartsTooltip formatter={(v: unknown) => `${(Number(v) || 0).toFixed(1)}%`} />
                <Bar dataKey="share" fill="#f59e0b" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* ROC */}
      <Card>
        <CardHeader><CardTitle>ROC Curves (one-vs-rest)</CardTitle></CardHeader>
        <CardContent className="h-72">
          <ResponsiveContainer>
            <LineChart margin={{ left: 12, right: 12, top: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="FPR" unit="%" domain={[0, 100]} />
              <YAxis type="number" dataKey="y" name="TPR" unit="%" domain={[0, 100]} />
              <RechartsTooltip />
              <Legend />
              {rocSeries.map((s, i) => (
                <Line
                  key={s.label}
                  data={s.points}
                  dataKey="y"
                  name={`${s.label}${s.auc != null ? ` (AUC ${s.auc.toFixed(3)})` : ""}`}
                  stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
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
              {prSeries.map((s, i) => (
                <Line
                  key={s.label}
                  data={s.points}
                  dataKey="y"
                  name={`${s.label}${s.ap != null ? ` (AP ${s.ap.toFixed(3)})` : ""}`}
                  stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Confusion Matrix */}
      <Card>
        <CardHeader><CardTitle>Confusion Matrix (test split)</CardTitle></CardHeader>
        <CardContent>
          <div className="overflow-auto">
            <table className="min-w-[480px] border-collapse">
              <thead>
                <tr>
                  <th className="p-2 text-left text-xs text-slate-400">True \ Pred</th>
                  {labels.map((l) => (
                    <th key={l} className="p-2 text-xs text-slate-400">{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cm.map((row, rIdx) => (
                  <tr key={rIdx}>
                    <td className="p-2 text-xs text-slate-400">{labels[rIdx] ?? `C${rIdx}`}</td>
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

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <Card>
      <CardHeader><CardTitle className="text-sm text-slate-400">{label}</CardTitle></CardHeader>
      <CardContent>
        <div className="text-2xl font-semibold">{value}</div>
        {sub && <div className="mt-1 text-xs text-slate-500">{sub}</div>}
      </CardContent>
    </Card>
  );
}
