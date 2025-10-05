"use client";

import React, { useState, useEffect } from "react";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getFeatureInfo, type FeatureInfo } from "@/lib/api";
import { cn } from "@/lib/utils";

type Values = Record<string, string | number>;

interface ClassifierFormProps {
  features: string[];
  values: Values;
  onChange: (values: Values) => void;
  onSubmit: () => void;
  onUseSample: () => void;
  loading?: boolean;
}

export function ClassifierForm({
  features,
  values,
  onChange,
  onSubmit,
  onUseSample,
  loading = false,
}: ClassifierFormProps) {
  const [featureInfo, setFeatureInfo] = useState<FeatureInfo>({});
  const [showUnits, setShowUnits] = useState(false);

  useEffect(() => {
    getFeatureInfo().then(setFeatureInfo).catch(console.error);
  }, []);

  const handleChange = (feature: string, value: string) => {
    onChange({ ...values, [feature]: value === "" ? "" : Number(value) });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Classifier Inputs</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={showUnits}
            onChange={(e) => setShowUnits(e.target.checked)}
            className="rounded border-slate-600 bg-slate-800 text-amber-500 focus:ring-amber-500"
          />
          Show units in placeholders
        </label>

        <TooltipProvider>
          <div className="grid gap-3 sm:grid-cols-2">
            {features.map((feature) => {
              const info = featureInfo[feature];
              const placeholder =
                showUnits && info?.unit ? `e.g., in ${info.unit}` : "Enter value";
              return (
                <div key={feature} className="group">
                  <div className="mb-1 flex items-center justify-between">
                    <span className="text-sm text-slate-300">{feature}</span>
                    {info && (
                      <Tooltip>
                        <TooltipTrigger className="text-slate-500 hover:text-slate-300">
                          <Info size={16} />
                        </TooltipTrigger>
                        <TooltipContent>
                          <div className="space-y-1">
                            {info.hint && <div>{info.hint}</div>}
                            {info.unit && <div className="text-slate-400">Unit: {info.unit}</div>}
                          </div>
                        </TooltipContent>
                      </Tooltip>
                    )}
                  </div>
                  <input
                    type="number"
                    step="any"
                    inputMode="decimal"
                    value={values[feature] ?? ""}
                    placeholder={placeholder}
                    onChange={(e) => handleChange(feature, e.target.value)}
                    className={cn(
                      "w-full px-3 py-2 rounded-lg",
                      "bg-slate-800/50 border border-slate-700",
                      "text-slate-200 placeholder-slate-500",
                      "focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500",
                      "transition-all duration-200",
                      "group-hover:border-slate-600"
                    )}
                  />
                </div>
              );
            })}
          </div>
        </TooltipProvider>

        <div className="flex gap-2 pt-2">
          <button
            type="button"
            onClick={onUseSample}
            className="rounded-xl border border-slate-700 bg-slate-800/60 px-4 py-2 text-sm hover:bg-slate-800"
          >
            Use Sample
          </button>
          <button
            type="button"
            onClick={onSubmit}
            disabled={loading}
            className="rounded-xl bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-60"
          >
            {loading ? "Analyzing..." : "Predict"}
          </button>
        </div>
      </CardContent>
    </Card>
  );
}
