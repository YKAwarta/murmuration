"use client";
import React, { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { toast } from "@/components/ui/toast";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { type DecisionStats, type PredictResponse } from "@/lib/api";
import { formatPercent } from "@/lib/utils";
import { Bird } from "lucide-react";

interface ResultPanelProps {
  result: PredictResponse | null;
  threshold: number;
  defaultThreshold: number;
  targetPrecision?: number;
  defaultStats?: DecisionStats;
  thresholdCurve?: DecisionStats[];
  onThresholdChange: (value: number) => void;
}

const SLIDER_MIN = 0.34; // with three classes the top probability is always >= 1/3
const SLIDER_MAX = 0.99;

export function ResultPanel({
  result,
  threshold,
  defaultThreshold,
  targetPrecision,
  defaultStats,
  thresholdCurve,
  onThresholdChange,
}: ResultPanelProps) {
  // Same rule as the API: accept when the top calibrated probability reaches the threshold
  const isAccepted = useMemo(
    () => (result ? result.decision.confidence >= threshold : false),
    [result, threshold]
  );

  const isDefault = Math.abs(threshold - defaultThreshold) < 1e-9;

  // Held-out test coverage/precision: exact at the default, else the closest evaluated threshold
  const atThreshold = useMemo(() => {
    if (isDefault && defaultStats) return defaultStats;
    if (!thresholdCurve?.length) return null;
    return thresholdCurve.reduce((best, s) =>
      Math.abs(s.threshold - threshold) < Math.abs(best.threshold - threshold) ? s : best
    );
  }, [isDefault, defaultStats, thresholdCurve, threshold]);

  const thresholdControl = (
    <div>
      <div className="mb-2 flex items-center justify-between text-sm text-slate-300">
        <span>Decision threshold: {threshold.toFixed(2)}</span>
        {isDefault ? (
          <span className="text-xs text-slate-500">default from calibration split</span>
        ) : (
          <button
            type="button"
            onClick={() => onThresholdChange(defaultThreshold)}
            className="rounded-md border border-slate-700 px-2 py-0.5 text-xs text-slate-300 hover:bg-slate-800"
          >
            Reset to default ({defaultThreshold.toFixed(2)})
          </button>
        )}
      </div>
      <Slider
        value={[threshold]}
        onValueChange={(v) => onThresholdChange(v[0] ?? threshold)}
        min={SLIDER_MIN}
        max={SLIDER_MAX}
        step={0.01}
        className="w-full"
      />
      <div className="mt-1 flex justify-between text-xs text-slate-500">
        <span>Permissive ({SLIDER_MIN.toFixed(2)})</span>
        <span>Strict ({SLIDER_MAX.toFixed(2)})</span>
      </div>
      {atThreshold && (
        <p className="mt-2 text-xs text-slate-400">
          On the held-out test split, this threshold accepts{" "}
          {formatPercent(atThreshold.coverage)} of objects
          {atThreshold.precision_on_accepted != null && (
            <> with {formatPercent(atThreshold.precision_on_accepted)} of them classified correctly</>
          )}
          .{targetPrecision != null && isDefault && (
            <> The default targets {formatPercent(targetPrecision, 0)} precision.</>
          )}
        </p>
      )}
    </div>
  );

  if (!result) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Prediction Result</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <p>Run a prediction to see results.</p>
          {thresholdControl}
        </CardContent>
      </Card>
    );
  }

  const copyToClipboard = () => {
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    toast({ message: "Result copied to clipboard", type: "success" });
  };

  const probData = Object.entries(result.probs).map(([label, prob]) => ({
    label,
    calibrated: prob * 100,
    uncalibrated: (result.probs_uncalibrated?.[label] ?? 0) * 100,
  }));

  const factors = result.top_factors.slice(0, 5);
  const maxAbs = Math.max(...factors.map((f) => Math.abs(f.shap)), 1e-9);

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle className="flex items-center gap-2">
          Prediction Result
          <Badge variant={isAccepted ? "accepted" : "uncertain"}>
            {isAccepted ? (
              <span className="inline-flex items-center gap-1">
                <Bird className="h-3.5 w-3.5" />
                Predicted: {result.label}
              </span>
            ) : (
              `Uncertain (${result.label}?) — needs review`
            )}
          </Badge>
        </CardTitle>
        <button
          onClick={copyToClipboard}
          className="rounded-lg border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:bg-slate-800"
        >
          Copy JSON
        </button>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3">
            <div className="text-xs text-slate-400">Confidence</div>
            <div className="text-xl font-semibold">{formatPercent(result.decision.confidence)}</div>
            <div className="text-[11px] text-slate-500 mt-0.5">Calibrated top-class probability</div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3">
            <div className="text-xs text-slate-400">Margin</div>
            <div className="text-xl font-semibold">{formatPercent(result.decision.margin)}</div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              Lead over {result.decision.second_best.label.toLowerCase()}
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3">
            <div className="text-xs text-slate-400">Decision</div>
            <div className="text-xl font-semibold">{isAccepted ? "Accept" : "Review"}</div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              {formatPercent(result.decision.confidence)} {isAccepted ? "≥" : "<"} {threshold.toFixed(2)}
            </div>
          </div>
        </div>

        {thresholdControl}

        <div className="space-y-2">
          <div className="text-sm font-medium">Class Probabilities</div>
          <div className="text-xs text-slate-500 -mt-1">
            Temperature-scaled (served) vs. the raw model output
          </div>
          <div className="h-56 w-full">
            <ResponsiveContainer>
              <BarChart data={probData} margin={{ left: 12, right: 12, top: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" />
                <YAxis unit="%" domain={[0, 100]} />
                <Tooltip formatter={(v: unknown) => `${(Number(v) || 0).toFixed(1)}%`} />
                <Legend />
                <Bar dataKey="calibrated" name="Calibrated" fill="#f59e0b" />
                <Bar dataKey="uncalibrated" name="Uncalibrated" fill="#475569" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-sm font-medium">Top Contributing Factors</div>
          <div className="text-xs text-slate-500 -mt-1 mb-2">
            TreeSHAP contribution of each input to the calibrated log-odds of {result.label}.
            Green pushes toward it, red away from it.
          </div>
          <div className="space-y-2">
            {factors.map((f) => (
              <div key={f.feature}>
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="text-slate-300">
                    {f.feature} <span className="text-slate-500">= {f.value ?? "missing"}</span>
                  </span>
                  <span>{f.shap >= 0 ? "+" : ""}{f.shap.toFixed(3)}</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded bg-slate-800">
                  <div
                    className={`h-2 ${f.shap >= 0 ? "bg-emerald-500" : "bg-red-500"}`}
                    style={{ width: `${(Math.abs(f.shap) / maxAbs) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
