"use client";
import React, { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { toast } from "@/components/ui/toast";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { type PredictResponse } from "@/lib/api";
import { formatPercent } from "@/lib/utils";
import { Bird } from "lucide-react";

interface ResultPanelProps {
  result: PredictResponse | null;
  threshold: number;
  onThresholdChange: (value: number) => void;
}

export function ResultPanel({ result, threshold, onThresholdChange }: ResultPanelProps) {
  const isAccepted = useMemo(
    () => (result ? result.decision.confidence >= threshold : false),
    [result, threshold]
  );

  if (!result) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Prediction Result</CardTitle>
        </CardHeader>
        <CardContent>Run a prediction to see results.</CardContent>
      </Card>
    );
  }

  const copyToClipboard = () => {
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    toast({ message: "Result copied to clipboard", type: "success" });
  };

  const probData = Object.entries(result.probs).map(([label, prob]) => ({
    label,
    probability: prob * 100,
  }));

  const factorData = result.top_factors.slice(0, 5).map((f) => ({
    feature: f.feature,
    importance: (f.shap ?? f.importance ?? 0) * 100,
    value: f.value,
  }));

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
      "Uncertain — Needs Review"
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
    <div className="text-[11px] text-slate-500 mt-0.5">Flock confidence</div>
  </div>
  <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3">
    <div className="text-xs text-slate-400">Margin</div>
    <div className="text-xl font-semibold">{formatPercent(result.decision.margin)}</div>
    <div className="text-[11px] text-slate-500 mt-0.5">Lead over next class</div>
  </div>
  <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-3">
    <div className="text-xs text-slate-400">Decision Threshold</div>
    <div className="text-sm">{threshold.toFixed(2)}</div>
    <div className="text-[11px] text-slate-500 mt-0.5">Gate to join the flock</div>
  </div>
</div>

        <div>
          <div className="mb-2 text-sm text-slate-300">Adjust Decision Threshold</div>
          <Slider
            value={[threshold]}
            onValueChange={(v) => onThresholdChange(v[0] ?? threshold)}
            min={0.1}
            max={0.95}
            step={0.01}
            className="w-full"
          />
          <div className="mt-1 flex justify-between text-xs text-slate-500">
            <span>Conservative (0.10)</span>
            <span>Strict (0.95)</span>
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-sm font-medium">Class Probabilities</div>
          <div className="h-56 w-full">
            <ResponsiveContainer>
              <BarChart data={probData} margin={{ left: 12, right: 12, top: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" />
                <YAxis unit="%" />
                <Tooltip />
                <Bar dataKey="probability">
                  {probData.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={
                        entry.probability >= 80
                          ? "#10b981"
                          : entry.probability >= 50
                          ? "#f59e0b"
                          : "#64748b"
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-sm font-medium">Top Contributing Factors</div>
          <div className="text-xs text-slate-500 -mt-1 mb-2">Why the flock turned</div>
          <div className="space-y-2">
            {factorData.map((f) => (
              <div key={f.feature}>
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="text-slate-300">{f.feature}</span>
                  <span>Value: {f.value ?? "null"}</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded bg-slate-800">
                  <div
                    className={`h-2 ${f.importance >= 0 ? "bg-emerald-500" : "bg-red-500"}`}
                    style={{ width: `${Math.min(100, Math.abs(f.importance))}%` }}
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
