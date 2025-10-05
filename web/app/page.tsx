"use client";
import { useEffect, useState } from "react";
import Image from "next/image";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ClassifierForm } from "@/components/ClassifierForm";
import { ResultPanel } from "@/components/ResultPanel";
import { MetricStrip } from "@/components/MetricStrip";
import Insights from "@/components/Insights";
import { toast } from "@/components/ui/toast";
import {
  getMetadata,
  getMetricsFull,
  getEchoSample,
  predict,
  type Metadata,
  type PredictResponse,
  type MetricsFull,
} from "@/lib/api";

export default function Home() {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [metricsFull, setMetricsFull] = useState<MetricsFull | null>(null);
  const [features, setFeatures] = useState<Record<string, string | number>>({});
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [threshold, setThreshold] = useState<number>(0.5);
  const [loading, setLoading] = useState(false);
  const [loadingMetrics, setLoadingMetrics] = useState(false);

  useEffect(() => {
    getMetadata()
      .then((data) => {
        setMetadata(data);
        setThreshold(data.metrics.recommended_threshold ?? 0.5);
        const empty: Record<string, string | number> = {};
        data.features.forEach((f) => (empty[f] = ""));
        setFeatures(empty);
      })
      .catch((err) => {
        console.error("Failed to load metadata:", err);
        toast({ message: "Failed to load metadata", type: "error" });
      });
  }, []);

  const handleTabChange = (value: string) => {
    if (value === "insights" && !metricsFull && !loadingMetrics) {
      setLoadingMetrics(true);
      getMetricsFull()
        .then(setMetricsFull)
        .catch((err) => {
          console.error("Failed to load metrics:", err);
          toast({ message: "Failed to load metrics", type: "error" });
        })
        .finally(() => setLoadingMetrics(false));
    }
  };

  const handleUseSample = async () => {
    try {
      const sample = await getEchoSample();
      const next: Record<string, string | number> = {};
      metadata?.features.forEach((f) => (next[f] = sample.features[f] ?? ""));
      setFeatures(next);
      toast({ message: "Sample data loaded", type: "success" });
      if (sample.true_label) {
        toast({ message: `True label: ${sample.true_label}`, type: "info" });
      }
    } catch (err) {
      console.error("Failed to load sample:", err);
      toast({ message: "Failed to load sample", type: "error" });
    }
  };

  const handlePredict = async () => {
    if (!metadata) return;
    const valid: Record<string, number> = {};
    Object.entries(features).forEach(([k, v]) => {
      if (v !== "" && !Number.isNaN(Number(v))) valid[k] = Number(v);
    });
    if (Object.keys(valid).length === 0) {
      toast({ message: "Please enter at least one feature value", type: "error" });
      return;
    }
    setLoading(true);
    try {
      const pred = await predict(valid);
      setResult(pred);
      toast({ message: "Prediction complete", type: "success" });
    } catch (e) {
      console.error("Prediction failed:", e);
      toast({ message: "Prediction failed", type: "error" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="relative mx-auto max-w-6xl px-4 py-10">
      {/* Animated starfield background */}
      <div className="stars" />
      <div className="stars2" />
      <div className="stars3" />

      {/* Hero */}
      <section className="mb-8 flex flex-col items-center text-center">
        <Image
          src="/murmuration_logo.png"
          alt="Murmuration"
          width={84}
          height={84}
          className="mb-2"
          priority
        />
        <h1 className="text-3xl font-bold">Murmuration</h1>
        <p className="text-slate-400">Exoplanet Classifier</p>
        <p className="mt-2 text-sm text-slate-400">
          Birds Beyond the Horizon • AI-powered classification using NASA Kepler/TESS data
        </p>
      </section>

      <Tabs defaultValue="predict" onValueChange={handleTabChange}>
        <TabsList>
          <TabsTrigger value="predict">Predict</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
        </TabsList>

        <TabsContent value="predict">
          <div className="grid gap-6 md:grid-cols-2">
            <ClassifierForm
              features={metadata?.features ?? []}
              values={features}
              onChange={setFeatures}
              onSubmit={handlePredict}
              onUseSample={handleUseSample}
              loading={loading}
            />
            <ResultPanel
              result={result}
              threshold={threshold}
              onThresholdChange={setThreshold}
            />
          </div>

          <MetricStrip metadata={metadata} />
        </TabsContent>

        <TabsContent value="insights">
          <Insights metrics={metricsFull} loading={loadingMetrics} />
        </TabsContent>
      </Tabs>
    </main>
  );
}
