export const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8080";

export type Metadata = {
  features: string[];
  labels: string[];
  metrics: { macro_f1: number; roc_auc_ovr: number; n_train: number; n_test: number };
  version: string;
};

export type PredictResponse = {
  label: string;
  probs: Record<string, number>;
  top_factors: { feature: string; importance: number; value: number | null }[];
};

export async function getMetadata(): Promise<Metadata> {
  const r = await fetch(`${API}/metadata`, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getSample(): Promise<{ features: Record<string, number | null>; true_label?: string }>{
  const r = await fetch(`${API}/echo-sample`, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function postPredict(features: Record<string, number>): Promise<PredictResponse> {
  const r = await fetch(`${API}/predict`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ features }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
