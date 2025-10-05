const API_URL = process.env.NEXT_PUBLIC_API_URL || "/api";

// --- Shared types ---
export type Probs = Record<string, number>;

export type RocCurve = {
  fpr: number[];
  tpr: number[];
  auc?: number;
};

export type PrCurve = {
  recall: number[];
  precision: number[];
  ap?: number;
};

export interface Metadata {
  features: string[];
  labels: string[];
  metrics: {
    macro_f1: number;
    roc_auc_ovr: number;
    n_train: number;
    n_test: number;
    recommended_threshold?: number;
  };
  version: string;
}

export interface PredictDecision {
  accepted: boolean;
  reason: string;
  threshold: number;
  confidence: number; // winning prob
  margin: number;     // winning - second_best
  second_best: { label: string; prob: number };
}

export interface PredictResponse {
  label: string;
  probs: Probs;
  decision: PredictDecision;
  top_factors: Array<{
    feature: string;
    shap?: number;
    importance?: number;
    value: number | null;
  }>;
}

export interface MetricsFull {
  confusion_matrix: number[][];
  roc: Record<string, RocCurve>;
  pr: Record<string, PrCurve>;
  ece: number;
  calibration_bins: {
    bin_mid: number[];
    acc: number[];
    conf: number[];
    count: number[];
  };
  auc_per_class: Record<string, number>;
  feature_importances_gain: Record<string, number>;
  feature_importances_shap?: Record<string, number>;
  recommended_threshold: number;
  top_confusions: Array<{ true: string; pred: string; count: number }>;
  search: {
    best_params: Record<string, unknown>;
    cv_macro_f1_mean: number;
    cv_macro_f1_std: number;
  };
}

export type FeatureInfo = {
  [key: string]: { unit: string; hint: string };
};

// --- Fetch helpers ---
export async function getMetadata(): Promise<Metadata> {
  const res = await fetch(`${API_URL}/metadata`);
  if (!res.ok) throw new Error("Failed to fetch metadata");
  return res.json();
}

export async function getMetricsFull(): Promise<MetricsFull> {
  const res = await fetch(`${API_URL}/metrics_full`);
  if (!res.ok) throw new Error("Failed to fetch metrics");
  return res.json();
}

export async function getFeatureInfo(): Promise<FeatureInfo> {
  const res = await fetch(`${API_URL}/feature_info`);
  if (!res.ok) throw new Error("Failed to fetch feature info");
  return res.json();
}

export async function getEchoSample(): Promise<{
  features: Record<string, number>;
  true_label?: string;
}> {
  const res = await fetch(`${API_URL}/echo-sample`);
  if (!res.ok) throw new Error("Failed to fetch sample");
  return res.json();
}

export async function predict(
  features: Record<string, number>
): Promise<PredictResponse> {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  if (!res.ok) throw new Error("Prediction failed");
  return res.json();
}

export async function batchPredict(
  rows: Array<{ features: Record<string, number> }>
): Promise<PredictResponse[]> {
  const res = await fetch(`${API_URL}/batch_predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(rows),
  });
  if (!res.ok) throw new Error("Batch prediction failed");
  return res.json();
}
