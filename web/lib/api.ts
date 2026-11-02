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
    accuracy: number;
    macro_f1: number;
    roc_auc_ovr: number;
    n_train: number;
    n_calibration: number;
    n_test: number;
    ece_uncalibrated: number;
    ece_calibrated: number;
    brier_uncalibrated: number;
    brier_calibrated: number;
    recommended_threshold: number;
    target_precision: number;
  };
  calibration: { method: string; temperature: number };
  data: { retrieved_utc?: string; labeled_rows: number };
  version: string;
}

export interface PredictDecision {
  accepted: boolean;
  reason: string;
  threshold: number;
  threshold_source: "default" | "request";
  confidence: number; // winning (calibrated) prob
  margin: number;     // winning - second_best
  second_best: { label: string; prob: number };
}

export interface Contribution {
  feature: string;
  value: number | null;
  shap: number; // contribution to the predicted class's calibrated logit
}

export interface PredictResponse {
  label: string;
  probs: Probs;              // temperature-scaled
  probs_uncalibrated: Probs; // raw LightGBM softmax
  decision: PredictDecision;
  attribution: {
    method: string;
    explains: string;
    class: string;
    base_value: number;
    logit: number;
    contributions: Contribution[];
  };
  top_factors: Contribution[];
}

export interface ReliabilityBins {
  bin_mid: number[];
  conf: number[];
  acc: number[];
  count: number[];
}

export interface CalibrationReport {
  log_loss: number;
  brier: number;
  ece_top_label: number;
  ece_classwise: number;
  reliability_bins: ReliabilityBins;
}

export interface DecisionStats {
  threshold: number;
  coverage: number;
  precision_on_accepted: number | null;
  n_accepted: number;
}

export interface MetricsFull {
  labels: string[];
  search: {
    best_params: Record<string, unknown>;
    cv_macro_f1_mean: number;
    cv_macro_f1_std: number;
  };
  test: {
    n: number;
    accuracy: number;
    macro_f1: number;
    roc_auc_ovr_macro: number;
    auc_per_class: Record<string, number>;
    confusion_matrix: number[][];
    top_confusions: Array<{ true: string; pred: string; count: number }>;
    roc: Record<string, RocCurve>;
    pr: Record<string, PrCurve>;
  };
  calibration: {
    method: string;
    temperature: number;
    test_before: CalibrationReport;
    test_after: CalibrationReport;
  };
  decision: {
    target_precision: number;
    recommended_threshold: number;
    test: DecisionStats;
    threshold_curve_test: DecisionStats[];
  };
  feature_importance: {
    mean_abs_shap: Record<string, number>;
    gain: Record<string, number>;
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

export async function getEchoSample(index = 0): Promise<{
  features: Record<string, number | null>;
  true_label?: string;
  mission?: string;
  index: number;
  n_samples: number;
}> {
  const res = await fetch(`${API_URL}/echo-sample?index=${index}`);
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

export interface BatchPrediction {
  label: string;
  probs: Probs;
  confidence: number;
  accepted: boolean;
}

export async function batchPredict(
  rows: Array<{ features: Record<string, number> }>
): Promise<BatchPrediction[]> {
  const res = await fetch(`${API_URL}/batch_predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(rows),
  });
  if (!res.ok) throw new Error("Batch prediction failed");
  return res.json();
}
