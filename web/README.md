# Murmuration web app

Next.js front end for the Murmuration API. Setup, scripts and screenshots are in the
[repository README](../README.md#frontend-nextjs).

The app calls `/api/*`, which `next.config.mjs` proxies to the API at `http://127.0.0.1:8081`
(to point it elsewhere, set `API_PROXY_TARGET` when running `npm run dev` or `npm run build`).
