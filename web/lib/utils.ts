import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(n: number, decimals = 2): string {
  return n.toFixed(decimals);
}

export function formatPercent(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

export function getConfidenceColor(prob: number): string {
  if (prob >= 0.8) return "text-emerald-400";
  if (prob >= 0.6) return "text-amber-400";
  return "text-red-400";
}

export function getDecisionColor(accepted: boolean): string {
  return accepted
    ? "bg-emerald-500/20 text-emerald-400 border-emerald-500"
    : "bg-amber-500/20 text-amber-400 border-amber-500";
}
