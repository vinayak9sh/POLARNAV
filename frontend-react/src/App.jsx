import "./App.css";
import AntarcticMap from "./components/AntarcticMap";
import NavigationGuidanceHUD from "./components/NavigationGuidanceHUD";
import { useEffect, useState, useRef } from "react";
import {
  getForecastGrid,
  getLongRangeForecastGrid,
  getIcebergRiskGrid,
  calculateRoute,
  replanRoute,
  getForecastWindow,
} from "./services/api";

// Haversine distance (km) between two lat/lon points
function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function getDistanceToRouteInKm(vesselPos, routeCoordinates) {
  if (!vesselPos || !routeCoordinates || routeCoordinates.length < 2) {
    return 0.0;
  }

  const vLat = Number(vesselPos.latitude);
  const vLon = Number(vesselPos.longitude);
  if (!Number.isFinite(vLat) || !Number.isFinite(vLon)) return 0.0;

  let minDistanceKm = Infinity;

  for (let i = 0; i < routeCoordinates.length - 1; i++) {
    const ptA = routeCoordinates[i];
    const ptB = routeCoordinates[i + 1];

    const aLon = Array.isArray(ptA) ? Number(ptA[0]) : Number(ptA.longitude);
    const aLat = Array.isArray(ptA) ? Number(ptA[1]) : Number(ptA.latitude);
    const bLon = Array.isArray(ptB) ? Number(ptB[0]) : Number(ptB.longitude);
    const bLat = Array.isArray(ptB) ? Number(ptB[1]) : Number(ptB.latitude);

    const cosLat = Math.cos((aLat * Math.PI) / 180.0);

    const dxV = (vLon - aLon) * 111.0 * cosLat;
    const dyV = (vLat - aLat) * 111.0;

    const dxB = (bLon - aLon) * 111.0 * cosLat;
    const dyB = (bLat - aLat) * 111.0;

    const lenSq = dxB * dxB + dyB * dyB;

    let dist = 0.0;
    if (lenSq === 0) {
      dist = Math.hypot(dxV, dyV);
    } else {
      const t = Math.max(0, Math.min(1, (dxV * dxB + dyV * dyB) / lenSq));
      const qx = t * dxB;
      const qy = t * dyB;
      dist = Math.hypot(dxV - qx, dyV - qy);
    }

    if (dist < minDistanceKm) {
      minDistanceKm = dist;
    }
  }

  return Number.isFinite(minDistanceKm) ? minDistanceKm : 0.0;
}

const OFF_TRACK_BUFFER_KM = 5;
const ARRIVAL_THRESHOLD_KM = 8; // km – geographic arrival radius for the simulation

function App() {
  const [forecastDate, setForecastDate] = useState("");
  const [forecastWindow, setForecastWindow] = useState(null);
  const [loadingForecastWindow, setLoadingForecastWindow] = useState(true);
  const [forecastWindowError, setForecastWindowError] = useState(null);

  const [forecastData, setForecastData] = useState(null);
  const [forecastMetadata, setForecastMetadata] = useState(null);

  const [loadingForecast, setLoadingForecast] = useState(false);

  const [forecastError, setForecastError] = useState(null);

  const [displayMode, setDisplayMode] = useState("sic");

  const [routeData, setRouteData] = useState(null);

  const [selectedRouteIndex, setSelectedRouteIndex] = useState(0);

  const [loadingRoute, setLoadingRoute] = useState(false);

  const [routeError, setRouteError] = useState(null);

  const [vesselPosition, setVesselPosition] = useState(null);
  const [vesselHeading, setVesselHeading] = useState(0);
  const [vesselSpeed, setVesselSpeed] = useState(0);
  const [offTrackDistance, setOffTrackDistance] = useState("0.0");
  const [isOffTrack, setIsOffTrack] = useState(false);
  const isAutoReplanningRef = useRef(false);

  const [replannedRouteData, setReplannedRouteData] = useState(null);

  const [loadingReplan, setLoadingReplan] = useState(false);

  const [replanError, setReplanError] = useState(null);

  const [icebergData, setIcebergData] = useState(null);

  const [loadingIcebergs, setLoadingIcebergs] = useState(false);

  const [icebergError, setIcebergError] = useState(null);

  // ── Voyage completion tracking ──────────────────────────────────────────────
  const [voyageComplete, setVoyageComplete] = useState(false);
  const [voyageStats, setVoyageStats] = useState(null);
  const voyageStartTimeRef = useRef(null);       // wall-clock ms when voyage first moved
  const voyageDistanceRef = useRef(0);           // accumulated km travelled
  const voyageReplansRef = useRef(0);            // dynamic replan count
  const prevPosRef = useRef(null);               // last position for step-Δdist calculation
  const voyageCompleteRef = useRef(false);       // ref mirror so interval closure is always fresh
  const voyageSessionRef = useRef(0);            // incremented on every new voyage / reset
  const lastValidRiskRef = useRef({ score: null, level: null }); // last non-null risk from active route

  // Track initial coords so "Plan New Voyage" can snap back to start
  const initialStartRef = useRef({ lat: -59.93, lon: 49.89 });

  useEffect(() => {
    let cancelled = false;

    async function loadForecastWindow() {
      setLoadingForecastWindow(true);
      setForecastWindowError(null);

      try {
        const data = await getForecastWindow();

        if (!cancelled) {
          setForecastWindow(data);

          // Start with the first valid forecast date:
          // the day immediately after the latest observation.
          const firstForecastDate = new Date(
            `${data.latest_observation_date}T12:00:00Z`
          );

          firstForecastDate.setUTCDate(
            firstForecastDate.getUTCDate() + 1
          );

          const firstValidForecastDate =
            firstForecastDate
              .toISOString()
              .split("T")[0];

          setForecastDate(
            firstValidForecastDate
          );
          setForecastData(null);
          setForecastMetadata(null);
        }
      } catch (error) {
        if (!cancelled) {
          setForecastWindowError(error.message);
        }
      } finally {
        if (!cancelled) {
          setLoadingForecastWindow(false);
        }
      }
    }

    loadForecastWindow();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadForecast() {
      setLoadingForecast(true);
      setForecastError(null);

      try {
        if (
          !forecastDate ||
          !forecastWindow ||
          forecastDate <= forecastWindow.latest_observation_date
        ) {
          return;
        }

        const data = await getLongRangeForecastGrid(
          forecastDate
        );

        if (!cancelled) {
          setForecastData(data);
          setForecastMetadata({
            forecast_date: data.forecast_date,
            latest_observation_date: data.latest_observation_date,
            horizon_months: data.horizon_months,
            method: data.method,
            confidence: data.confidence,
            uncertainty: data.uncertainty,
          });
        }
      } catch (error) {
        if (!cancelled) {
          setForecastError(error.message);
        }
      } finally {
        if (!cancelled) {
          setLoadingForecast(false);
        }
      }
    }

    loadForecast();

    return () => {
      cancelled = true;
    };
  }, [forecastDate, forecastWindow]);

  useEffect(() => {
    if (displayMode !== "icebergs") {
      return;
    }

    let cancelled = false;

    async function loadIcebergs() {
      setLoadingIcebergs(true);
      setIcebergError(null);

      try {
        const data = await getIcebergRiskGrid(forecastDate);

        if (!cancelled) {
          setIcebergData(data);
        }
      } catch (error) {
        if (!cancelled) {
          setIcebergError(error.message);
        }
      } finally {
        if (!cancelled) {
          setLoadingIcebergs(false);
        }
      }
    }

    loadIcebergs();

    return () => {
      cancelled = true;
    };
  }, [displayMode, forecastDate]);

  async function handleCalculateRoute() {
    setLoadingRoute(true);
    setRouteError(null);
    setSelectedRouteIndex(0);
    setReplannedRouteData(null);

    // Bump session ID so any in-flight replan from a previous voyage is ignored
    voyageSessionRef.current += 1;

    // Reset all voyage tracking state
    setVoyageComplete(false);
    setVoyageStats(null);
    voyageStartTimeRef.current = null;
    voyageDistanceRef.current = 0;
    voyageReplansRef.current = 0;
    prevPosRef.current = null;
    voyageCompleteRef.current = false;
    lastValidRiskRef.current = { score: null, level: null };

    const startLat = Number(document.getElementById("startLatitude").value);
    const startLon = Number(document.getElementById("startLongitude").value);

    // Remember start so "Plan New Voyage" can snap back
    initialStartRef.current = { lat: startLat, lon: startLon };

    try {
      const data = await calculateRoute({
        forecast_date: forecastDate,

        vessel_profile: document.getElementById("vesselProfile").value,

        start_latitude: startLat,

        start_longitude: startLon,

        destination_latitude: Number(
          document.getElementById("destinationLatitude").value,
        ),

        destination_longitude: Number(
          document.getElementById("destinationLongitude").value,
        ),
      });

      setRouteData(data);
      setVesselPosition({ latitude: startLat, longitude: startLon });
      setVesselSpeed(0);
    } catch (error) {
      setRouteError(error.message);
    } finally {
      setLoadingRoute(false);
    }
  }

  function handleSimulateProgress() {
    if (
      !routeData?.geometry?.coordinates ||
      routeData.geometry.coordinates.length < 3
    ) {
      return;
    }

    const coordinates = routeData.geometry.coordinates;

    const midpointIndex = Math.floor(coordinates.length / 2);

    const midpoint = coordinates[midpointIndex];

    // GeoJSON format:
    // [longitude, latitude]
    const longitude = Number(midpoint[0]);

    const latitude = Number(midpoint[1]);

    setVesselPosition({
      latitude,
      longitude,
    });
  }

  async function handleReplanRouteFromPos(pos) {
    const targetPos = pos || vesselPosition;
    if (!targetPos) return;

    // Snapshot the session at request-time; discard if the voyage has since reset
    const sessionAtCall = voyageSessionRef.current;

    setLoadingReplan(true);
    setReplanError(null);

    try {
      const selectedR = routeData?.alternative_routes
        ? routeData.alternative_routes[selectedRouteIndex]
        : routeData;
      const category =
        selectedR?.category ||
        (selectedRouteIndex === 0
          ? "safest"
          : selectedRouteIndex === 1
            ? "efficient"
            : "balanced");

      const data = await replanRoute({
        forecast_date: forecastDate,

        vessel_profile:
          document.getElementById("vesselProfile")?.value || "standard",

        current_latitude: targetPos.latitude,

        current_longitude: targetPos.longitude,

        destination_latitude: Number(
          document.getElementById("destinationLatitude").value,
        ),

        destination_longitude: Number(
          document.getElementById("destinationLongitude").value,
        ),

        category: category,
      });

      // Only apply if we are still in the same voyage session
      if (sessionAtCall !== voyageSessionRef.current) return;

      setReplannedRouteData(data);
      voyageReplansRef.current += 1;
    } catch (error) {
      if (sessionAtCall === voyageSessionRef.current) {
        setReplanError(error.message);
      }
    } finally {
      if (sessionAtCall === voyageSessionRef.current) {
        setLoadingReplan(false);
      }
    }
  }

  async function handleReplanRoute() {
    return handleReplanRouteFromPos(vesselPosition);
  }

  // Keep lastValidRiskRef up-to-date whenever the active route changes
  useEffect(() => {
    const activeR = replannedRouteData
      ? replannedRouteData
      : routeData?.alternative_routes
        ? (routeData.alternative_routes[selectedRouteIndex] || routeData.alternative_routes[0])
        : routeData;
    if (!activeR) return;
    const score = activeR?.risk_score ?? null;
    const level = activeR?.risk_level ?? null;
    if (score !== null && score !== 0) {
      lastValidRiskRef.current = { score, level };
    }
  }, [routeData, replannedRouteData, selectedRouteIndex]);

  // Real-time movement effect: updates vessel position continuously along heading when speed > 0
  useEffect(() => {
    if (vesselSpeed <= 0 || !vesselPosition) return;

    // Record voyage start time on first movement
    if (!voyageStartTimeRef.current) {
      voyageStartTimeRef.current = Date.now();
    }

    const destLat = Number(document.getElementById("destinationLatitude")?.value ?? NaN);
    const destLon = Number(document.getElementById("destinationLongitude")?.value ?? NaN);
    const hasValidDest = Number.isFinite(destLat) && Number.isFinite(destLon);

    const interval = setInterval(() => {
      if (voyageCompleteRef.current) return;

      setVesselPosition((prevPos) => {
        if (!prevPos) return prevPos;

        const stepDistKm = vesselSpeed * 0.005;
        const rad = (vesselHeading * Math.PI) / 180.0;

        const deltaLat = (stepDistKm / 111.0) * Math.cos(rad);
        const cosLat = Math.max(
          0.1,
          Math.cos((prevPos.latitude * Math.PI) / 180.0),
        );
        const deltaLon = (stepDistKm / (111.0 * cosLat)) * Math.sin(rad);

        const newPos = {
          latitude: prevPos.latitude + deltaLat,
          longitude: prevPos.longitude + deltaLon,
        };

        // Accumulate distance travelled
        if (prevPosRef.current) {
          const stepKm = haversineKm(
            prevPosRef.current.latitude, prevPosRef.current.longitude,
            newPos.latitude, newPos.longitude,
          );
          voyageDistanceRef.current += stepKm;
        }
        prevPosRef.current = newPos;

        // Arrival check
        if (hasValidDest) {
          const distToDest = haversineKm(newPos.latitude, newPos.longitude, destLat, destLon);
          if (distToDest <= ARRIVAL_THRESHOLD_KM && !voyageCompleteRef.current) {
            voyageCompleteRef.current = true;
            const elapsedMs = voyageStartTimeRef.current
              ? Date.now() - voyageStartTimeRef.current
              : 0;
            // Use the last valid non-zero risk captured before arrival
            const { score: riskScore, level: riskLevel } = lastValidRiskRef.current;
            const activeR = routeData?.alternative_routes
              ? (routeData.alternative_routes[selectedRouteIndex] || routeData.alternative_routes[0])
              : routeData;
            const plannedDistKm = activeR?.route?.distance_km ?? null;
            setVoyageStats({
              distanceTravelledKm: voyageDistanceRef.current,
              plannedDistanceKm: plannedDistKm,
              remainingKm: 0,          // Fix #1: always 0 at arrival
              elapsedMs,
              dynamicReplans: voyageReplansRef.current,
              finalRiskScore: riskScore, // Fix #2: last valid non-zero risk
              finalRiskLevel: riskLevel,
              routeName: activeR?.label || activeR?.route_name || "Selected Route",
              destLat,
              destLon,
            });
            setVoyageComplete(true);
            // Snap vessel to destination so map marker lands exactly there
            return { latitude: destLat, longitude: destLon };
          }
        }

        return newPos;
      });
    }, 100);

    return () => clearInterval(interval);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vesselSpeed, vesselHeading, vesselPosition !== null, routeData, selectedRouteIndex]);

  // Off-track buffer monitoring effect: checks distance to route & triggers auto-replan if > OFF_TRACK_BUFFER_KM
  useEffect(() => {
    // Do not replan after arrival
    if (!vesselPosition || voyageCompleteRef.current) return;

    const baseRoute = replannedRouteData || routeData;
    const activeR =
      Array.isArray(baseRoute?.alternative_routes) &&
      baseRoute.alternative_routes.length > 0
        ? baseRoute.alternative_routes[selectedRouteIndex] ||
          baseRoute.alternative_routes[0]
        : baseRoute;
    const coords = activeR?.geometry?.coordinates || activeR?.coordinates;

    if (!coords || coords.length < 2) return;

    const dist = getDistanceToRouteInKm(vesselPosition, coords);
    setOffTrackDistance(dist.toFixed(1));

    const isExceeded = dist > OFF_TRACK_BUFFER_KM;
    setIsOffTrack(isExceeded);

    if (isExceeded && !loadingReplan && !isAutoReplanningRef.current) {
      isAutoReplanningRef.current = true;
      handleReplanRouteFromPos(vesselPosition).finally(() => {
        setTimeout(() => {
          isAutoReplanningRef.current = false;
        }, 3000);
      });
    }
  }, [
    vesselPosition,
    routeData,
    replannedRouteData,
    selectedRouteIndex,
    loadingReplan,
    voyageComplete,   // re-evaluate when arrival is detected
  ]);


  // "Plan New Voyage" performs a complete application page refresh
  function handleDismissVoyage() {
    window.location.reload();
  }

  // Format elapsed ms into h m s string
  function formatElapsed(ms) {
    if (!ms || ms <= 0) return "0s";
    const totalSec = Math.floor(ms / 1000);
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = totalSec % 60;
    const parts = [];
    if (h > 0) parts.push(`${h}h`);
    if (m > 0) parts.push(`${m}m`);
    parts.push(`${s}s`);
    return parts.join(" ");
  }

  return (
    <div className="app">
      {/* ------------------------------------------------
          Top navigation
      ------------------------------------------------ */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="brand-icon-wrap" aria-hidden="true">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L15 9H22L16.5 13.5L18.5 20.5L12 16L5.5 20.5L7.5 13.5L2 9H9L12 2Z" fill="url(#brandGrad)" />
              <circle cx="12" cy="12" r="3" fill="#ffffff" />
              <defs>
                <linearGradient id="brandGrad" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#00f0ff" />
                  <stop offset="1" stopColor="#0066ff" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div>
            <div className="brand-title-line">
              <span className="brand-primary">POLARNAV</span>
              <span className="brand-divider">/</span>
              <h1 className="brand-subtitle-title">Antarctic Navigation</h1>
            </div>
            <p className="topbar-desc">Predictive Maritime Decision-Support System</p>
          </div>
        </div>

        <div className="system-status">
          <span className="status-pulse-ring">
            <span className="status-dot"></span>
          </span>
          <div className="system-status-text">
            <span className="status-label">SYSTEM READY</span>
            <span className="status-detail">Predictive Engine Online</span>
          </div>
        </div>
      </header>

      {/* ------------------------------------------------
          Main application
      ------------------------------------------------ */}
      <main className="workspace">
        {/* Left control panel */}
        <aside className="sidebar">
          <section className="panel">
            <h2>Voyage Planning</h2>

            <label htmlFor="forecastDate">Forecast Date</label>

            <input
              id="forecastDate"
              type="date"
              value={forecastDate}
              min={
                forecastWindow
                  ? (() => {
                      const date = new Date(
                        `${forecastWindow.latest_observation_date}T00:00:00`
                      );
                      date.setDate(date.getDate() + 1);
                      return date.toISOString().slice(0, 10);
                    })()
                  : ""
              }
              max={
                forecastWindow
                  ? forecastWindow.maximum_forecast_date
                  : ""
              }
              onChange={(event) =>
                setForecastDate(event.target.value)
              }
              disabled={
                loadingForecastWindow ||
                Boolean(forecastWindowError)
              }
            />

            {loadingForecastWindow && (
              <div className="navigation-status">
                Loading forecast window...
              </div>
            )}

            {forecastWindowError && (
              <div className="navigation-status">
                Forecast window error: {forecastWindowError}
              </div>
            )}

            {forecastWindow && (
              <div className="navigation-status">
                Select a forecast date between{" "}
                {forecastWindow.latest_observation_date} and{" "}
                {forecastWindow.maximum_forecast_date}.
              </div>
            )}

            {forecastMetadata && (
              <div className="navigation-status forecast-intelligence">
                <strong>Forecast Intelligence</strong>

                <div>
                  Horizon:{" "}
                  {forecastMetadata.horizon_months} month
                  {forecastMetadata.horizon_months !== 1 ? "s" : ""}
                </div>

                <div>
                  Method:{" "}
                  {forecastMetadata.method === "anomaly_aware"
                    ? "Anomaly-aware"
                    : "Seasonal climatology"}
                </div>

                <div>
                  Confidence:{" "}
                  <strong>
                    {forecastMetadata.confidence}
                  </strong>
                </div>

                <div>
                  Expected error: ±
                  {forecastMetadata.uncertainty
                    ?.typical_p75_percent_sic}%
                  SIC
                </div>

                <div>
                  Conservative error: ±
                  {forecastMetadata.uncertainty
                    ?.conservative_p90_percent_sic}%
                  SIC
                </div>
              </div>
            )}

            <label htmlFor="vesselProfile">Vessel Profile</label>

            <select id="vesselProfile" defaultValue="standard">
              <option value="conservative">Conservative</option>

              <option value="standard">Standard</option>

              <option value="ice_capable">Ice Capable</option>
            </select>

            <label htmlFor="startLatitude">Start Latitude</label>

            <input
              id="startLatitude"
              type="number"
              defaultValue="-59.93"
              step="0.0001"
            />

            <label htmlFor="startLongitude">Start Longitude</label>

            <input
              id="startLongitude"
              type="number"
              defaultValue="49.89"
              step="0.0001"
            />

            <label htmlFor="destinationLatitude">Destination Latitude</label>

            <input
              id="destinationLatitude"
              type="number"
              defaultValue="-59.93"
              step="0.0001"
            />

            <label htmlFor="destinationLongitude">Destination Longitude</label>

            <input
              id="destinationLongitude"
              type="number"
              defaultValue="68.56"
              step="0.0001"
            />

            <button
              className="primary-button"
              onClick={handleCalculateRoute}
              disabled={loadingRoute}
            >
              {loadingRoute ? "Calculating..." : "Calculate Route"}
            </button>

            {routeError && (
              <div className="navigation-status">Route error: {routeError}</div>
            )}

            {routeError && (
              <div className="navigation-status">Route error: {routeError}</div>
            )}

            {routeData?.alternative_routes && routeData.alternative_routes.length > 0 && (
              <div className="navigation-status">
                <strong>Navigation Choice Matrix</strong>
                <p style={{ margin: "4px 0 10px 0", color: "#8ea3b7", fontSize: "11px" }}>
                  Select a route corridor to preview metrics and map path:
                </p>
                <div className="route-selector-cards">
                  {routeData.alternative_routes.map((r, idx) => {
                    const isSelected = selectedRouteIndex === idx;
                    const cat = r.category || (idx === 0 ? "safest" : idx === 1 ? "efficient" : "balanced");
                    const icon = cat.includes("safest") ? "🛡️" : cat.includes("efficient") ? "⚡" : "⚖️";

                    return (
                      <div
                        key={idx}
                        className={`route-card ${isSelected ? "active" : ""} category-${cat}`}
                        onClick={() => setSelectedRouteIndex(idx)}
                      >
                        <div className="route-card-header">
                          <span className="route-card-icon">{icon}</span>
                          <div style={{ flex: 1 }}>
                            <strong className="route-card-title">{r.label || r.route_name}</strong>
                            <div className="route-card-tagline">{r.tagline || "Navigation Choice"}</div>
                          </div>
                        </div>

                        <div className="route-card-risk-summary">
                          <span className="risk-score-label">
                            <strong>Risk Score:</strong> {r.risk_score !== undefined ? Number(r.risk_score).toFixed(1) : "0.0"} / 100
                          </span>
                          <span className="risk-sep">·</span>
                          <span className={`risk-badge risk-${(r.risk_level || "LOW").toLowerCase()}`}>
                            {r.risk_level || "LOW"}
                          </span>
                        </div>

                        <div className="route-card-metrics">
                          <div className="metric-item">
                            <span className="metric-label">Distance</span>
                            <span className="metric-val">{r.route?.distance_km} km</span>
                          </div>
                          <div className="metric-item">
                            <span className="metric-label">Nav Cost</span>
                            <span className="metric-val">{r.route?.total_navigation_cost}</span>
                          </div>
                        </div>

                        {r.best_for && (
                          <div className="route-card-best-for">
                            <strong>Best for:</strong> {r.best_for}
                          </div>
                        )}

                        {r.tradeoffs && (
                          <div className="route-card-tradeoffs">
                            {r.tradeoffs.advantages?.map((adv, aIdx) => (
                              <div key={`adv-${aIdx}`} className="tradeoff-pill advantage">
                                + {adv}
                              </div>
                            ))}
                            {r.tradeoffs.disadvantages?.map((dis, dIdx) => (
                              <div key={`dis-${dIdx}`} className="tradeoff-pill disadvantage">
                                - {dis}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {routeData && (
              <div className="navigation-status">
                <strong>Selected Choice ({routeData.alternative_routes ? routeData.alternative_routes[selectedRouteIndex]?.label || routeData.alternative_routes[selectedRouteIndex]?.route_name || "Optimal" : "Primary"})</strong>

                {(() => {
                  const currentR = routeData.alternative_routes
                    ? routeData.alternative_routes[selectedRouteIndex]
                    : routeData;
                  return (
                    <>
                      <div>Distance: {currentR.route.distance_km} km</div>
                      <div>Navigation cost: {currentR.route.total_navigation_cost}</div>
                      {currentR.risk_score !== undefined && (
                        <div>Environmental Risk Score: {Number(currentR.risk_score).toFixed(1)} / 100 ({currentR.risk_level})</div>
                      )}
                    </>
                  );
                })()}

                {routeData.forecast && (
                  <>
                    <hr />

                    <strong>Forecast used</strong>

                    <div>
                      Date:{" "}
                      {String(
                        routeData.forecast.forecast_date
                      ).slice(0, 10)}
                    </div>

                    <div>
                      Horizon:{" "}
                      {routeData.forecast.horizon_months} month
                      {routeData.forecast.horizon_months !== 1
                        ? "s"
                        : ""}
                    </div>

                    <div>
                      Method:{" "}
                      {routeData.forecast.method ===
                      "anomaly_aware"
                        ? "Anomaly-aware"
                        : "Seasonal climatology"}
                    </div>

                    <div>
                      Confidence:{" "}
                      <strong>
                        {routeData.forecast.confidence}
                      </strong>
                    </div>
                  </>
                )}
              </div>
            )}
          </section>

          {/* Environmental layers */}
          <section className="panel">
            <h2>Environment</h2>

            <div className="layer-buttons">
              <button
                className={`layer-button ${
                  displayMode === "sic" ? "active" : ""
                }`}
                onClick={() => setDisplayMode("sic")}
              >
                Predicted SIC
              </button>

              <button
                className={`layer-button ${
                  displayMode === "risk" ? "active" : ""
                }`}
                onClick={() => setDisplayMode("risk")}
              >
                Risk Level
              </button>
            </div>
          </section>

          {loadingForecast && (
            <div className="navigation-status">
              Loading predicted sea-ice conditions...
            </div>
          )}

          {forecastError && (
            <div className="navigation-status">
              Forecast error: {forecastError}
            </div>
          )}

          {/* Dynamic navigation */}
          <section className="panel">
            <h2>Vessel Navigation Controls</h2>

            {/* Ship Heading & Rotation Controls */}
            <div className="vessel-control-group">
              <label>Vessel Heading: <strong>{vesselHeading}°</strong></label>
              <div className="rotation-button-row">
                <button
                  type="button"
                  className="rotate-btn"
                  onClick={() => setVesselHeading((h) => (h - 15 + 360) % 360)}
                >
                  ↺ Rotate -15°
                </button>
                <button
                  type="button"
                  className="rotate-btn"
                  onClick={() => setVesselHeading((h) => (h + 15) % 360)}
                >
                  ↻ Rotate +15°
                </button>
              </div>
            </div>

            {/* Speed Control Slider */}
            <div className="vessel-control-group">
              <label>
                Vessel Speed: <strong>{vesselSpeed === 0 ? "0 (Stopped)" : `${vesselSpeed} knots`}</strong>
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={vesselSpeed}
                onChange={(e) => setVesselSpeed(Number(e.target.value))}
                className="speed-slider"
              />
            </div>

            {/* Off-track Buffer & Telemetry */}
            {vesselPosition && (
              <div className="navigation-status telemetry-box">
                <div><strong>Distance to Route:</strong> {offTrackDistance} km</div>
                <div><strong>Buffer Threshold:</strong> {OFF_TRACK_BUFFER_KM.toFixed(1)} km</div>
                <div style={{ marginTop: "6px" }}>
                  <span className={`status-badge ${isOffTrack ? "off-track" : "on-track"}`}>
                    {isOffTrack ? "⚠️ OFF TRACK (Rerouting...)" : "✅ ON TRACK (Within Buffer)"}
                  </span>
                </div>
              </div>
            )}

            {replanError && (
              <div className="navigation-status">
                Replan error: {replanError}
              </div>
            )}

            {routeData && (
              <div className="navigation-status route-summary">
                {(() => {
                  const activeR = replannedRouteData ||
                    (routeData?.alternative_routes ? routeData.alternative_routes[selectedRouteIndex] : routeData);
                  return (
                    <>
                      <strong>{replannedRouteData ? "Replanned Route" : activeR?.route_name || "Selected Route"}</strong>

                      <div>Distance: {activeR.route.distance_km} km</div>

                      <div>Cost: {activeR.route.total_navigation_cost}</div>
                    </>
                  );
                })()}

                {replannedRouteData && (
                  <>
                    <hr />

                    <strong>Replanned Route</strong>

                    <div>
                      Remaining: {replannedRouteData.route.distance_km} km
                    </div>

                    <div>
                      Cost: {replannedRouteData.route.total_navigation_cost}
                    </div>

                    <hr />

                    <strong>Route Change</strong>

                    <div>
                      Distance change:{" "}
                      {(
                        replannedRouteData.route.distance_km -
                        routeData.route.distance_km
                      ).toFixed(2)}{" "}
                      km
                    </div>

                    <div>
                      Distance change:{" "}
                      {(
                        ((replannedRouteData.route.distance_km -
                          routeData.route.distance_km) /
                          routeData.route.distance_km) *
                        100
                      ).toFixed(1)}
                      %
                    </div>

                    <div>
                      Navigation cost change:{" "}
                      {(
                        replannedRouteData.route.total_navigation_cost -
                        routeData.route.total_navigation_cost
                      ).toFixed(2)}
                    </div>
                  </>
                )}
              </div>
            )}

            {(routeData?.decision || replannedRouteData?.decision) && (
              <div className="navigation-status decision-summary">
                <strong>Decision Intelligence</strong>

                {(() => {
                  const activeR = replannedRouteData ||
                    (routeData?.alternative_routes ? routeData.alternative_routes[selectedRouteIndex] : routeData);
                  const decision = activeR?.decision;

                  if (!decision) return <div>No decision breakdown available.</div>;

                  const seaIceContribution =
                    decision.contribution_percent?.sea_ice ?? 0;

                  const icebergContribution =
                    decision.contribution_percent?.iceberg ?? 0;

                  return (
                    <>
                      <div>
                        Primary factor:{" "}
                        {decision.dominant_hazard === "sea_ice"
                          ? "Sea Ice"
                          : decision.dominant_hazard === "iceberg"
                            ? "Icebergs"
                            : "Combined"}
                      </div>

                      <div>
                        Sea-ice contribution: {seaIceContribution.toFixed(1)}%
                      </div>

                      <div>
                        Iceberg contribution: {icebergContribution.toFixed(1)}%
                      </div>

                      <div>
                        Iceberg-affected route cells:{" "}
                        {decision.iceberg_exposure?.affected_route_cells ?? 0}
                      </div>

                      <div>
                        Maximum iceberg penalty:{" "}
                        {decision.iceberg_exposure?.maximum_penalty ?? 0}
                      </div>
                    </>
                  );
                })()}
              </div>
            )}

            <div className="navigation-status">
              {replannedRouteData
                ? "Route successfully replanned from the vessel's current position."
                : vesselPosition
                  ? "Vessel position updated. Ready to replan."
                  : routeData
                    ? "Initial route calculated. Select alternative corridors or simulate vessel progress."
                    : "Ready for route planning."}
            </div>
          </section>
        </aside>

        {/* Map area */}
        <section className="map-container">
          <div id="map" className="map">
            <AntarcticMap
              forecastData={forecastData}
              icebergData={icebergData}
              routeData={replannedRouteData || routeData}
              vesselPosition={vesselPosition}
              vesselHeading={vesselHeading}
              routeIsReplanned={Boolean(replannedRouteData)}
              displayMode={displayMode}
              selectedRouteIndex={selectedRouteIndex}
              onSelectRoute={setSelectedRouteIndex}
            />
          </div>

          {/* Turn-by-Turn Navigation Guidance HUD (Top Right) */}
          <NavigationGuidanceHUD
            routeData={routeData}
            replannedRouteData={replannedRouteData}
            vesselPosition={vesselPosition}
            vesselHeading={vesselHeading}
            vesselSpeed={vesselSpeed}
            selectedRouteIndex={selectedRouteIndex}
          />

          {/* Decision panel */}
          <div className="decision-panel">
            {/* Navigation status */}
            <div>
              <span className="metric-label">Navigation</span>

              <strong>
                {replannedRouteData
                  ? "REPLANNED ROUTE"
                  : routeData
                    ? (routeData.alternative_routes?.[selectedRouteIndex]?.route_name?.toUpperCase() || "PRIMARY ROUTE")
                    : "AWAITING ROUTE"}
              </strong>
            </div>

            {/* Distance */}
            <div>
              <span className="metric-label">Distance</span>

              <strong>
                {replannedRouteData
                  ? `${replannedRouteData.route.distance_km} km remaining`
                  : routeData
                    ? `${(routeData.alternative_routes?.[selectedRouteIndex] || routeData).route.distance_km} km`
                    : "—"}
              </strong>
            </div>

            {/* Navigation cost */}
            <div>
              <span className="metric-label">Navigation Cost</span>

              <strong>
                {replannedRouteData
                  ? replannedRouteData.route.total_navigation_cost
                  : routeData
                    ? (routeData.alternative_routes?.[selectedRouteIndex] || routeData).route.total_navigation_cost
                    : "—"}
              </strong>
            </div>

            {/* Decision factor */}
            <div>
              <span className="metric-label">Primary Factor</span>

              <strong>
                {(() => {
                  const activeDec = (replannedRouteData || (routeData?.alternative_routes?.[selectedRouteIndex] || routeData))?.decision;
                  if (!activeDec) return "—";
                  return activeDec.dominant_hazard === "sea_ice"
                    ? "Sea Ice"
                    : activeDec.dominant_hazard === "iceberg"
                      ? "Icebergs"
                      : "Combined";
                })()}
              </strong>
            </div>
          </div>
        </section>
      </main>

      {/* ── Voyage Complete Modal ──────────────────────────────── */}
      {voyageComplete && voyageStats && (
        <div className="voyage-overlay" role="dialog" aria-modal="true" aria-label="Voyage Complete">
          <div className="voyage-modal">
            <div className="voyage-modal-header">
              <span className="voyage-modal-icon">🏁</span>
              <div>
                <h2 className="voyage-modal-title">Destination Reached</h2>
                <p className="voyage-modal-subtitle">{voyageStats.routeName}</p>
              </div>
            </div>

            <div className="voyage-stats-grid">
              <div className="voyage-stat-card">
                <span className="voyage-stat-label">Distance Travelled</span>
                <span className="voyage-stat-value">{voyageStats.distanceTravelledKm.toFixed(1)} <em>km</em></span>
              </div>

              <div className="voyage-stat-card">
                <span className="voyage-stat-label">Planned Route</span>
                <span className="voyage-stat-value">
                  {voyageStats.plannedDistanceKm !== null ? `${voyageStats.plannedDistanceKm} km` : "—"}
                </span>
              </div>

              <div className="voyage-stat-card">
                <span className="voyage-stat-label">Remaining at Arrival</span>
                <span className="voyage-stat-value voyage-stat-green">{voyageStats.remainingKm.toFixed(2)} <em>km</em></span>
              </div>

              <div className="voyage-stat-card">
                <span className="voyage-stat-label">Voyage Duration</span>
                <span className="voyage-stat-value">{formatElapsed(voyageStats.elapsedMs)}</span>
              </div>

              <div className="voyage-stat-card">
                <span className="voyage-stat-label">Dynamic Reroutes</span>
                <span className="voyage-stat-value">{voyageStats.dynamicReplans}</span>
              </div>

              <div className="voyage-stat-card">
                <span className="voyage-stat-label">Final Risk Score</span>
                <span className="voyage-stat-value">
                  {voyageStats.finalRiskScore !== null
                    ? <>{Number(voyageStats.finalRiskScore).toFixed(1)} <em>/ 100</em> &nbsp;<span className={`risk-badge risk-${(voyageStats.finalRiskLevel || "low").toLowerCase()}`}>{voyageStats.finalRiskLevel || "—"}</span></>
                    : "—"}
                </span>
              </div>
            </div>

            <div className="voyage-modal-actions">
              <button
                type="button"
                className="voyage-dismiss-btn"
                onMouseDown={() => window.location.reload()}
              >
                Plan New Voyage
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
