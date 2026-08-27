# Docker

Local development

```bash
npm install
npm run dev
```

Docker build and run

```bash
docker build -t ragify-frontend --build-arg VITE_API_URL=http://ragify-api:8000 .

docker run --rm -p 3000:80 ragify-frontend
```

Docker Compose

```bash
docker compose up --build
```

Setting VITE_API_URL in Compose
- Use the `build.args.VITE_API_URL` value to point at the API service, for example `http://ragify-api:8000`.
