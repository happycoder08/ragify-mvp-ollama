# Vite Proxy Configuration

## Overview

The Vite dev server now proxies all `/api` requests to the backend server with full SSE streaming support.

## Configuration

### [vite.config.ts](vite.config.ts)
- Proxies `/api/*` to `VITE_API_BASE_URL` (default: `http://localhost:8000`)
- Disables buffering for `/api/query` to support SSE streaming
- Sets `x-accel-buffering: no` header for streaming responses

### Environment Variables

**.env** (development default)
```env
# Leave empty to use relative URLs with Vite proxy
VITE_API_BASE_URL=
```

**.env.local** (optional local override)
```env
# Backend API base URL for proxy
VITE_API_BASE_URL=http://localhost:8000
```

**.env.production** (production deployment)
```env
# Absolute URL for production API
VITE_API_BASE_URL=https://api.ragify.example.com
```

## How It Works

### Development Mode (with proxy)
1. Frontend runs on `http://localhost:5173` (Vite default)
2. Backend runs on `http://localhost:8000`
3. Frontend makes requests to `/api/login`, `/api/query`, etc.
4. Vite proxy forwards to `http://localhost:8000/api/...`
5. No CORS issues, seamless SSE streaming

### Production Mode (direct API)
1. Set `VITE_API_BASE_URL=https://api.yourserver.com`
2. Frontend makes requests to `https://api.yourserver.com/api/...`
3. Backend must handle CORS headers

## SSE Streaming Support

The proxy configuration includes special handling for `/api/query`:
- Disables response buffering
- Sets `x-accel-buffering: no` header
- Allows real-time token streaming without delays

## API Client Updates

Both [src/api.ts](src/api.ts) and [src/sse.ts](src/sse.ts) now:
- Use `VITE_API_BASE_URL` environment variable
- Default to empty string (relative URLs for proxy)
- Support absolute URLs for production

## Usage

### Start Development Server
```bash
npm run dev
```

The Vite dev server will:
- Run on `http://localhost:5173`
- Proxy `/api` requests to `http://localhost:8000`
- Support SSE streaming without buffering

### Environment-Specific Configuration

Create `.env.local` to override defaults without committing changes:
```bash
# Use different backend URL
VITE_API_BASE_URL=http://192.168.1.100:8000
```

## Testing SSE Streaming

1. Start backend: `http://localhost:8000`
2. Start frontend: `npm run dev` → `http://localhost:5173`
3. Login and navigate to `/query`
4. Ask a question and watch tokens stream in real-time
5. Network tab should show `text/event-stream` response type

## Troubleshooting

**SSE buffering/delayed streaming:**
- Verify proxy configure callback is working
- Check browser Network tab for `x-accel-buffering: no` header
- Ensure backend sends `Content-Type: text/event-stream`

**CORS errors:**
- In dev mode with proxy: shouldn't happen
- In production: backend must set CORS headers

**Connection refused:**
- Ensure backend is running on correct port
- Check `VITE_API_BASE_URL` matches backend URL
