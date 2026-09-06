const API_URL = "http://127.0.0.1:8000"

async function request(endpoint, options = {}) {
  const response = await fetch(
    `${API_URL}${endpoint}`,
    {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    }
  )

  let data = null

  try {
    data = await response.json()
  } catch {
    data = null
  }

  if (!response.ok) {
    const message =
      data?.detail ||
      `Request failed with status ${response.status}`

    throw new Error(message)
  }

  return data
}

export async function getForecastGrid(forecastDate) {
  return request("/forecast-grid", {
    method: "POST",
    body: JSON.stringify({
      forecast_date: forecastDate,
    }),
  })
}

export async function getIcebergRiskGrid(forecastDate) {
  return request("/iceberg-risk-grid", {
    method: "POST",
    body: JSON.stringify({
      forecast_date: forecastDate,
    }),
  })
}

export async function getIcebergList() {
  return request("/icebergs", {
    method: "GET",
  })
}

export async function calculateRoute(payload) {
  return request("/route", {
    method: "POST",
    body: JSON.stringify(payload),
  })
}

export async function replanRoute(payload) {
  return request("/replan", {
    method: "POST",
    body: JSON.stringify(payload),
  })
}