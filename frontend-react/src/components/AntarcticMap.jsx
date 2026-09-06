import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet.heat";

function riskColor(code) {
  if (code === 0) return "#2ca25f";
  if (code === 1) return "#f1c40f";
  if (code === 2) return "#e67e22";
  if (code === 3) return "#d73027";

  return null;
}

function icebergColor(prediction) {
  return prediction?.prediction_method === "ml_random_forest"
    ? "#8e44ad"
    : "#3498db";
}

function icebergMethodLabel(prediction) {
  return prediction?.prediction_method === "ml_random_forest"
    ? "ML prediction"
    : "Persistence estimate";
}

function AntarcticMap({
  forecastData,
  icebergData,
  routeData,
  vesselPosition,
  routeIsReplanned = false,
  displayMode = "sic",
}) {
  const mapRef = useRef(null);
  const containerRef = useRef(null);

  const sicLayerRef = useRef(null);
  const riskCanvasRef = useRef(null);
  const icebergLayerRef = useRef(null);
  const routeLayerRef = useRef(null);
  const vesselMarkerRef = useRef(null);

  // ------------------------------------------------------------
  // Create Leaflet map once
  // ------------------------------------------------------------

  useEffect(() => {
    if (!containerRef.current || mapRef.current) {
      return;
    }

    const map = L.map(containerRef.current, {
      zoomControl: true,
      attributionControl: true,
    });

    map.setView([-45, 0], 1.5);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
      maxZoom: 10,
    }).addTo(map);

    mapRef.current = map;

    setTimeout(() => {
      map.invalidateSize();
    }, 0);

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // ------------------------------------------------------------
  // Predicted SIC
  // ------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;

    if (!map || !forecastData?.points) {
      return;
    }

    if (sicLayerRef.current) {
      map.removeLayer(sicLayerRef.current);

      sicLayerRef.current = null;
    }

    if (displayMode !== "sic") {
      return;
    }

    const sicPoints = forecastData.points.map((point) => [
      point.latitude,
      point.longitude,
      Math.max(0.01, Number(point.sic) / 100),
    ]);

    const sicLayer = L.heatLayer(sicPoints, {
      radius: 22,
      blur: 18,
      maxZoom: 6,
      minOpacity: 0.25,
    });

    sicLayer.addTo(map);

    sicLayerRef.current = sicLayer;

    return () => {
      if (map.hasLayer(sicLayer)) {
        map.removeLayer(sicLayer);
      }
    };
  }, [forecastData, displayMode]);

  // ------------------------------------------------------------
  // Risk canvas
  // ------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;

    if (!map || !forecastData?.points) {
      return;
    }

    if (riskCanvasRef.current) {
      riskCanvasRef.current.remove();
      riskCanvasRef.current = null;
    }

    if (displayMode !== "risk") {
      return;
    }

    const canvas = document.createElement("canvas");

    canvas.style.position = "absolute";
    canvas.style.pointerEvents = "none";
    canvas.style.zIndex = "300";

    const pane = map.getPanes().overlayPane;

    pane.appendChild(canvas);

    riskCanvasRef.current = canvas;

    function renderRisk() {
      const size = map.getSize();

      const topLeft = map.containerPointToLayerPoint([0, 0]);

      L.DomUtil.setPosition(canvas, topLeft);

      canvas.width = Math.max(1, Math.floor(size.x));

      canvas.height = Math.max(1, Math.floor(size.y));

      const context = canvas.getContext("2d");

      if (!context) {
        return;
      }

      context.clearRect(0, 0, canvas.width, canvas.height);

      for (const point of forecastData.points) {
        const color = riskColor(Number(point.risk_code));

        if (!color) {
          continue;
        }

        const pixel = map.latLngToContainerPoint([
          Number(point.latitude),
          Number(point.longitude),
        ]);

        if (
          pixel.x < 0 ||
          pixel.y < 0 ||
          pixel.x > canvas.width ||
          pixel.y > canvas.height
        ) {
          continue;
        }

        context.fillStyle = color;

        context.fillRect(
          Math.floor(pixel.x) - 2,
          Math.floor(pixel.y) - 2,
          4,
          4,
        );
      }
    }

    renderRisk();

    map.on("moveend zoomend resize", renderRisk);

    return () => {
      map.off("moveend zoomend resize", renderRisk);

      canvas.remove();

      if (riskCanvasRef.current === canvas) {
        riskCanvasRef.current = null;
      }
    };
  }, [forecastData, displayMode]);

  // ------------------------------------------------------------
  // Iceberg layer
  // ------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;

    if (!map) {
      return;
    }

    // Remove previous iceberg layer
    if (icebergLayerRef.current) {
      map.removeLayer(icebergLayerRef.current);

      icebergLayerRef.current = null;
    }

    if (displayMode !== "icebergs" || !icebergData) {
      return;
    }

    /*
      The API currently returns the represented
      iceberg predictions as "predictions".
    */

    const predictions = Array.isArray(icebergData.predicted_icebergs)
      ? icebergData.predicted_icebergs
      : [];

    const icebergLayer = L.layerGroup();

    for (const prediction of predictions) {
      const latitude = Number(prediction.predicted_latitude);

      const longitude = Number(prediction.predicted_longitude);

      if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
        continue;
      }

      const marker = L.circleMarker([latitude, longitude], {
        radius: 7,
        color: "#ffffff",
        weight: 1.5,
        fillColor: icebergColor(prediction),
        fillOpacity: 0.9,
      });

      const method = icebergMethodLabel(prediction);

      const confidence = prediction.confidence || "unknown";

      const distance = Number(prediction.predicted_distance_km);

      const bearing = Number(prediction.predicted_bearing_deg);

      const age = prediction.observation_age_days;

      marker.bindPopup(`
        <div style="min-width: 190px">

          <b>Iceberg ${prediction.iceberg_id}</b>
          <br><br>

          <b>Method:</b>
          ${method}
          <br>

          <b>Confidence:</b>
          ${confidence}
          <br>

          <b>Prediction date:</b>
          ${prediction.prediction_date}
          <br>

          <b>Current position:</b>
          ${Number(prediction.current_latitude).toFixed(4)},
          ${Number(prediction.current_longitude).toFixed(4)}
          <br>

          <b>Predicted position:</b>
          ${latitude.toFixed(4)},
          ${longitude.toFixed(4)}
          <br>

          <b>Predicted movement:</b>
          ${Number.isFinite(distance) ? distance.toFixed(2) : "—"} km
          <br>

          <b>Bearing:</b>
          ${Number.isFinite(bearing) ? bearing.toFixed(1) : "—"}°
          ${
            age !== undefined ? `<br><b>Observation age:</b> ${age} day(s)` : ""
          }

        </div>
      `);

      icebergLayer.addLayer(marker);
    }

    icebergLayer.addTo(map);

    icebergLayerRef.current = icebergLayer;

    return () => {
      if (map.hasLayer(icebergLayer)) {
        map.removeLayer(icebergLayer);
      }
    };
  }, [icebergData, displayMode]);

  // ------------------------------------------------------------
  // Navigation route
  // ------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;

    if (!map) {
      return;
    }

    if (routeLayerRef.current) {
      map.removeLayer(routeLayerRef.current);

      routeLayerRef.current = null;
    }

    if (
      !routeData?.geometry?.coordinates ||
      routeData.geometry.coordinates.length < 2
    ) {
      return;
    }

    const coordinates = routeData.geometry.coordinates.map((point) => [
      Number(point[1]),
      Number(point[0]),
    ]);

    const route = L.polyline(coordinates, {
      color: routeIsReplanned ? "#8e44ad" : "#0066ff",
      weight: 5,
      opacity: 0.95,
      lineCap: "round",
      lineJoin: "round",
    });

    const start = L.circleMarker(coordinates[0], {
      radius: 7,
      color: "#ffffff",
      weight: 2,
      fillColor: "#0066ff",
      fillOpacity: 1,
    });

    start.bindPopup("<b>Route Start</b>");

    const destination = L.circleMarker(coordinates[coordinates.length - 1], {
      radius: 7,
      color: "#ffffff",
      weight: 2,
      fillColor: "#d73027",
      fillOpacity: 1,
    });

    destination.bindPopup("<b>Destination</b>");

    const routeGroup = L.layerGroup([route, start, destination]);

    routeGroup.addTo(map);

    routeLayerRef.current = routeGroup;

    console.log(
      routeIsReplanned
        ? "Displaying replanned route"
        : "Displaying initial route",
    );

    map.fitBounds(route.getBounds(), {
      padding: [50, 50],
      maxZoom: 5,
    });

    return () => {
      if (map.hasLayer(routeGroup)) {
        map.removeLayer(routeGroup);
      }

      if (routeLayerRef.current === routeGroup) {
        routeLayerRef.current = null;
      }
    };
  }, [routeData, routeIsReplanned]);

  // ------------------------------------------------------------
  // Current vessel position
  // ------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;

    if (!map) {
      return;
    }

    // Remove previous vessel marker
    if (vesselMarkerRef.current) {
      map.removeLayer(vesselMarkerRef.current);

      vesselMarkerRef.current = null;
    }

    if (
      !vesselPosition ||
      !Number.isFinite(Number(vesselPosition.latitude)) ||
      !Number.isFinite(Number(vesselPosition.longitude))
    ) {
      return;
    }

    const latitude = Number(vesselPosition.latitude);

    const longitude = Number(vesselPosition.longitude);

    const vesselMarker = L.circleMarker([latitude, longitude], {
      radius: 8,
      color: "#ffffff",
      weight: 2,
      fillColor: "#2c7fb8",
      fillOpacity: 1,
    });

    vesselMarker.bindPopup(`
    <b>Current Vessel Position</b><br>
    Latitude: ${latitude.toFixed(4)}<br>
    Longitude: ${longitude.toFixed(4)}
  `);

    vesselMarker.addTo(map);

    vesselMarkerRef.current = vesselMarker;

    vesselMarker.bringToFront();

    return () => {
      if (map.hasLayer(vesselMarker)) {
        map.removeLayer(vesselMarker);
      }

      if (vesselMarkerRef.current === vesselMarker) {
        vesselMarkerRef.current = null;
      }
    };
  }, [vesselPosition]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
      }}
    />
  );
}

export default AntarcticMap;
