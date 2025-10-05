export function toast({
  message,
  type = "info",
}: {
  message: string;
  type?: "success" | "error" | "info";
}) {
  const div = document.createElement("div");
  div.className = [
    "fixed bottom-4 right-4 z-50 px-4 py-2 rounded-lg text-sm font-medium animate-in",
    "slide-in-from-bottom-2",
    type === "success"
      ? "bg-emerald-500 text-white"
      : type === "error"
      ? "bg-red-500 text-white"
      : "bg-slate-700 text-slate-200",
  ].join(" ");
  div.textContent = message;
  document.body.appendChild(div);
  setTimeout(() => div.remove(), 3000);
}
