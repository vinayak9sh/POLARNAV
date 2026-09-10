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

export async function getForecastWindow() {
  const response = await fetch(
    "http://127.0.0.1:8000/forecast-window"
  );

  if (!response.ok) {
    throw new Error(
      `Forecast window request failed: ${response.status}`
    );
  }

  return response.json();
}

export async function getLongRangeForecastGrid(forecastDate) {
  const response = await fetch(
    "http://127.0.0.1:8000/forecast-long-range-grid",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        forecast_date: forecastDate,
      }),
    }
  );

  if (!response.ok) {
    let message = `Long-range forecast request failed: ${response.status}`;

    try {
      const errorData = await response.json();

      if (errorData?.detail) {
        message = errorData.detail;
      }
    } catch {
      // Keep the default error message.
    }

    throw new Error(message);
  }

  return response.json();
}