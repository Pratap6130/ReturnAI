# Return Risk Prediction

Predict whether an order is likely to be returned using a trained ML model, a FastAPI backend, and a React frontend.

## Highlights
- End-to-end pipeline: training -> model -> API -> UI
- FastAPI backend with auth, predictions, history, and health checks
- React UI for login, prediction form, history, and analytics
- Render-ready deployment (backend + static frontend)

## Tech Stack
- Python, FastAPI, scikit-learn, pandas, joblib
- React, Vite, Axios, Zustand, Recharts
- SQLite for local persistence

## Quick Start (Local)

### 1) Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2) Frontend
```bash
cd frontend-react
npm install
npm run dev
```

### 3) Open the App
- Frontend: http://localhost:5173
- Backend health: http://localhost:8000/health

## Environment Variables

### Backend
Create `backend/.env` (or set in your shell):
```
FRONTEND_URL=http://localhost:5173
```

### Frontend
Create `frontend-react/.env`:
```
VITE_API_URL=http://localhost:8000
```

## API Endpoints
- `GET /health`
- `POST /register`
- `POST /login`
- `POST /predict`
- `GET /predictions/recent?limit=10`

## Dataset and Model
- Trained model expected at `models/model.pkl`.
- Large CSV files are kept in `archive/` (one oversized file is ignored).

## Deployment (Render)

### Backend (Web Service)
- Build Command: `pip install -r backend/requirements.txt`
- Start Command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
- Env Var: `FRONTEND_URL=https://<your-frontend>.onrender.com`

### Frontend (Static Site)
- Root Directory: `frontend-react`
- Build Command: `npm install && npm run build`
- Publish Directory: `dist`
- Env Var: `VITE_API_URL=https://<your-backend>.onrender.com`
- Rewrite Rule: `/* -> /index.html`

See `DEPLOYMENT.md` for a full step-by-step guide.

## Project Structure
```
backend/        FastAPI app and SQLite storage
frontend-react/ React UI (Vite)
models/         Trained model artifacts
archive/        CSV source data
plots/          EDA images
src/            Training and data prep scripts
```
