/** @type {import('next').NextConfig} */
// The browser calls /api/*, which Next proxies to the FastAPI server.
const apiTarget = process.env.API_PROXY_TARGET || "http://127.0.0.1:8081";

const nextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiTarget}/:path*` }];
  },
};
export default nextConfig;
