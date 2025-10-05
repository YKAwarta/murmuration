"use client";

import { useEffect, useMemo, useState } from "react";
import { getMetadata, getSample, postPredict, type Metadata, type PredictResponse } from "@/lib/api";
import { Bird, Crosshair, Telescope, XCircle } from "lucide-react";

type Features = Record<string, number | "">;

const brand = {
  bg: "bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950",
  card: "bg-slate-900/60 border border-slate-800 rounded-2xl shadow-lg",
  accent: "text-amber-400",
  btn: "px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-900 font-semibold transition",
  btnGhost: "px-4 py-2 rounded-xl border border-slate-700 hover:bg-slate-800 transition",
  input: "w-full px-3 py-2 rounded-lg bg-slate-900/70 border border-slate-800 text-slate-100",
  label: "text-slate-300 text-sm",
  h1: "text-3xl md:text-4xl font-semibold text-slate-100",
  h2: "text-xl font-semibold text-slate-100",
  p: "text-slate-300",
};

export default function Home() {
  const [meta, setMeta] = useState<Metadata | null>(null);
  const [features, setFeatures] = useState<Features>({});
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState<number>(0.5);
  const [trueLabel, setTrueLabel] = useState<string | undefined>(undefined);

  // Load metadata on mount
  useEffect(() => {
    getMetadata().then((m) => {
      setMeta(m);
      const f: Features = {};
      m.features.forEach((k) => (f[k] = ""));
      setFeatures(f);
    }).catch((e) => setError(String(e)));
  }, []);

  const onUseSample = async () => {
    setError(null);
    try {
      const sample = await getSample();
      const merged: Features = {};
      meta?.features.forEach((k) => {
        const v = sample.features?.[k];
        merged[k] = (v === null || v === undefined) ? "" : Number(v);
      });
      setFeatures(merged);
      setTrueLabel(sample.true_label);
      setRes(null);
    } catch (e: any) { setError(String(e)); }
  };

  const onPredict = async () => {
    if (!meta) return;
    setLoading(true); setError(null); setRes(null);
    try {
      const payload: Record<string, number> = {};
      meta.features.forEach((k) => {
        const val = features[k];
        if (val !== "" && !Number.isNaN(Number(val))) payload[k] = Number(val);
      });
      const r = await postPredict(payload);
      setRes(r);
    } catch (e: any) { setError(String(e)); }
    finally { setLoading(false); }
  };

  const thresholdedLabel = useMemo(() => {
    if (!res) return null;
    let best = { label: "", p: -1 };
    Object.entries(res.probs).forEach(([lab, p]) => {
      if (p > best.p) best = { label: lab, p };
    });
    return best.p >= threshold ? best.label : "UNCERTAIN";
  }, [res, threshold]);

  return (
    <main className={`${brand.bg} min-h-screen text-slate-200`}>
      <div className="mx-auto max-w-6xl px-4 py-10 space-y-10">

        {/* Hero */}
        <section className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="space-y-2">
            <h1 className={brand.h1}>
              <span className={brand.accent}>Starling</span> — Birds Beyond the Horizon
            </h1>
            <p className={brand.p}>
              An elegant, bird-themed AI that classifies exoplanet candidates using NASA Kepler/TESS data.
            </p>
            <p className="text-slate-400 text-sm">
              API: {process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8080"}
            </p>
          </div>
          <div className="flex gap-3">
            <button className={brand.btn} onClick={onUseSample}><Telescope className="inline mr-2 h-4 w-4" />Use sample</button>
            <button className={brand.btnGhost} onClick={onPredict} disabled={loading}>
              {loading ? "Predicting..." : <>Predict <Crosshair className="inline ml-2 h-4 w-4" /></>}
            </button>
          </div>
        </section>

        {/* Body */}
        <section className="grid md:grid-cols-5 gap-6">
          {/* Feature form */}
          <div className={`md:col-span-3 p-5 ${brand.card}`}>
            <h2 className={brand.h2}><Bird className="inline mr-2" /> Classifier Inputs</h2>
            {!meta && <p className="mt-2 text-slate-400">Loading features…</p>}
            {meta && (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
                {meta.features.map((f) => (
                  <div key={f} className="space-y-1">
                    <label className={brand.label}>{f}</label>
                    <input
                      className={brand.input}
                      type="number"
                      step="any"
                      placeholder="(optional)"
                      value={features[f] ?? ""}
                      onChange={(e) => setFeatures({ ...features, [f]: e.target.value === "" ? "" : Number(e.target.value) })}
                    />
                  </div>
                ))}
              </div>
            )}
            <div className="flex items-center gap-4 mt-6">
              <button className={brand.btn} onClick={onPredict} disabled={loading || !meta}>
                {loading ? "Predicting..." : "Predict"}
              </button>
              {trueLabel && <span className="text-slate-400 text-sm">Sample true label: <em>{trueLabel}</em></span>}
              {error && <span className="text-red-400 text-sm">Error: {error}</span>}
            </div>
          </div>

          {/* Results */}
          <div className={`md:col-span-2 p-5 ${brand.card}`}>
            <h2 className={brand.h2}>Result</h2>

            {!res && <p className="mt-2 text-slate-400">Run a prediction to see results.</p>}

            {res && (
              <div className="space-y-5 mt-3">
                <div className="flex items-center gap-3">
                  <Badge label={thresholdedLabel || ""} />
                  <div className="flex-1">
                    <label className={brand.label}>Decision threshold: {threshold.toFixed(2)}</label>
                    <input
                      type="range"
                      min={0.1}
                      max={0.9}
                      step={0.01}
                      value={threshold}
                      onChange={(e) => setThreshold(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>

                <div>
                  <h3 className="text-slate-200 font-semibold mb-2">Class probabilities</h3>
                  <div className="space-y-2">
                    {Object.entries(res.probs).map(([lab, p]) => (
                      <ProbRow key={lab} label={lab} p={p} />
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="text-slate-200 font-semibold mb-2">Top factors</h3>
                  <ul className="divide-y divide-slate-800">
                    {res.top_factors.map((t) => (
                      <li key={t.feature} className="py-1 flex items-center justify-between">
                        <span className="text-slate-300">{t.feature}</span>
                        <span className="text-slate-400 text-sm">value: {t.value ?? "—"}</span>
                        <span className="text-slate-400 text-sm">importance: {Math.round(t.importance)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Metrics */}
        <section className={`p-5 ${brand.card}`}>
          <h2 className={brand.h2}>Model Metrics</h2>
          {!meta && <p className="mt-2 text-slate-400">Loading…</p>}
          {meta && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
              <Metric title="Macro F1" value={meta.metrics.macro_f1.toFixed(3)} />
              <Metric title="AUC (OvR)" value={Number.isFinite(meta.metrics.roc_auc_ovr) ? meta.metrics.roc_auc_ovr.toFixed(3) : "—"} />
              <Metric title="Train" value={`${meta.metrics.n_train}`} />
              <Metric title="Test" value={`${meta.metrics.n_test}`} />
            </div>
          )}
          <p className="text-slate-400 text-xs mt-3">
            Powered by NASA Kepler/TESS data. No logos used. AI-assisted build. Version {meta?.version ?? "—"}.
          </p>
        </section>

      </div>
    </main>
  );
}

function Metric({ title, value }: { title: string; value: string }) {
  return (
    <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
      <div className="text-slate-400 text-sm">{title}</div>
      <div className="text-slate-100 text-xl font-semibold">{value}</div>
    </div>
  );
}

function ProbRow({ label, p }: { label: string; p: number }) {
  const pct = Math.round(p * 100);
  return (
    <div className="w-full">
      <div className="flex justify-between text-sm text-slate-300">
        <span>{label}</span><span>{pct}%</span>
      </div>
      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
        <div className="h-full bg-amber-500" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function Badge({ label }: { label: string }) {
  const base = "px-3 py-1 rounded-full text-sm font-medium";
  if (label === "CONFIRMED") return <span className={`${base} bg-emerald-500/20 text-emerald-300 border border-emerald-600/40`}><Bird className="inline h-4 w-4 mr-1" /> Confirmed</span>;
  if (label === "CANDIDATE") return <span className={`${base} bg-sky-500/20 text-sky-300 border border-sky-600/40`}><Telescope className="inline h-4 w-4 mr-1" /> Candidate</span>;
  if (label === "FALSE POSITIVE") return <span className={`${base} bg-rose-500/20 text-rose-300 border border-rose-600/40`}><XCircle className="inline h-4 w-4 mr-1" /> False Positive</span>;
  return <span className={`${base} bg-slate-700/40 text-slate-300 border border-slate-600/40`}>Uncertain</span>;
}
