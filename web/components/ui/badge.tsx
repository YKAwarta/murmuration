import * as React from "react";
import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "accepted" | "uncertain" | "rejected";
}

export function Badge({
  className,
  variant = "default",
  children,
  ...props
}: BadgeProps) {
  const variants: Record<NonNullable<BadgeProps["variant"]>, string> = {
    default: "bg-slate-700 text-slate-200 border-slate-600",
    accepted:
      "bg-emerald-500/20 text-emerald-400 border-emerald-500 shadow-emerald-500/25",
    uncertain:
      "bg-amber-500/20 text-amber-400 border-amber-500 shadow-amber-500/25",
    rejected: "bg-red-500/20 text-red-400 border-red-500 shadow-red-500/25",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium",
        variants[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}
