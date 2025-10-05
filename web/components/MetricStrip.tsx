"use client";
import React from "react";
import { type Metadata } from "@/lib/api";

interface MetricStripProps {
  metadata: Metadata | null;
}

export function MetricStrip({ metadata }: MetricStripProps) {
  if (!metadata) return null;
  const { metrics, version } = metadata;
  return (
    <div className="mt-6 grid grid-cols-2 gap-2 text-sm text-slate-300 sm:grid-cols-5">
      <div>Macro F1: <span className="font-semibold">{metrics.macro_f1.toFixed(3)}</span></div>
      <div>AUC-OVR: <span className="font-semibold">{metrics.roc_auc_ovr.toFixed(3)}</span></div>
      <div>Train: <span className="font-semibold">{metrics.n_train.toLocaleString()}</span></div>
      <div>Test: <span className="font-semibold">{metrics.n_test.toLocaleString()}</span></div>
      <div>Version: <span className="font-semibold">{version}</span></div>
    </div>
  );
}
