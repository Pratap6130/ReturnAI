import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000",
  headers: { "Content-Type": "application/json" },
  timeout: 15000,
});

export async function loginUser(userid, password) {
  const { data } = await api.post("/login", { userid, password });
  return data;
}

export async function registerUser(name, userid, password) {
  const { data } = await api.post("/register", { name, userid, password });
  return data;
}

export async function predictReturn(features) {
  const { data } = await api.post("/predict", features);
  return data;
}

export async function getRecentPredictions(limit = 10) {
  const { data } = await api.get("/predictions/recent", { params: { limit } });
  return data;
}

export async function checkHealth() {
  const { data } = await api.get("/health");
  return data;
}

export default api;
