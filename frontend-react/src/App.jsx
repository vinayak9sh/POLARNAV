import "./App.css";
import AntarcticMap from "./components/AntarcticMap";
import { useEffect, useState } from "react";
import {
  getForecastGrid,
  getLongRangeForecastGrid,
  getIcebergRiskGrid,
  calculateRoute,
  replanRoute,
  getForecastWindow,
} from "./services/api";

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

  const [replannedRouteData, setReplannedRouteData] = useState(null);

  const [loadingReplan, setLoadingReplan] = useState(false);

  const [replanError, setReplanError] = useState(null);

  const [icebergData, setIcebergData] = useState(null);

  const [loadingIcebergs, setLoadingIcebergs] = useState(false);

  const [icebergError, setIcebergError] = useState(null);

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

    try {
      const data = await calculateRoute({
        forecast_date: forecastDate,

        vessel_profile: document.getElementById("vesselProfile").value,

        start_latitude: Number(document.getElementById("startLatitude").value),

        start_longitude: Number(
          document.getElementById("startLongitude").value,
        ),

        destination_latitude: Number(
          document.getElementById("destinationLatitude").value,
        ),

        destination_longitude: Number(
          document.getElementById("destinationLongitude").value,
        ),
      });

      setRouteData(data);
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

  async function handleReplanRoute() {
    if (!vesselPosition) {
      setReplanError("Simulate vessel progress first.");
      return;
    }

    setLoadingReplan(true);
    setReplanError(null);

    try {
      const data = await replanRoute({
        forecast_date: forecastDate,

        vessel_profile: document.getElementById("vesselProfile").value,

        current_latitude: vesselPosition.latitude,

        current_longitude: vesselPosition.longitude,

        destination_latitude: Number(
          document.getElementById("destinationLatitude").value,
        ),

        destination_longitude: Number(
          document.getElementById("destinationLongitude").value,
        ),
      });

      setReplannedRouteData(data);
    } catch (error) {
      setReplanError(error.message);
    } finally {
      setLoadingReplan(false);
    }
  }

  return (
    <div className="app">
      {/* ------------------------------------------------
          Top navigation
      ------------------------------------------------ */}
      <header className="topbar">
        <div>
          <h1>Antarctic Navigation</h1>
          <p>Predictive Maritime Decision Support</p>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          System Ready
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

            {routeData?.alternative_routes && routeData.alternative_routes.length > 1 && (
              <div className="navigation-status">
                <strong>Multiple Routes Generated</strong>
                <p style={{ margin: "4px 0 8px 0", color: "#8ea3b7" }}>
                  Select a route corridor to preview metrics and map path:
                </p>
                <div className="route-selector-cards">
                  {routeData.alternative_routes.map((r, idx) => {
                    const isSelected = selectedRouteIndex === idx;
                    const isPrimary = idx === 0;
                    const primaryDist = routeData.alternative_routes[0].route.distance_km;
                    const primaryCost = routeData.alternative_routes[0].route.total_navigation_cost;
                    const distDelta = r.route.distance_km - primaryDist;
                    const costDelta = r.route.total_navigation_cost - primaryCost;

                    return (
                      <div
                        key={idx}
                        className={`route-card ${isSelected ? "active" : ""}`}
                        onClick={() => setSelectedRouteIndex(idx)}
                      >
                        <div className="route-card-header">
                          <span className={`route-badge route-color-${idx}`}></span>
                          <strong>{r.route_name || (isPrimary ? "Primary (Optimal)" : `Alternative ${idx}`)}</strong>
                        </div>
                        <div className="route-card-metrics">
                          <span>{r.route.distance_km} km</span>
                          <span>Cost: {r.route.total_navigation_cost}</span>
                        </div>
                        {!isPrimary && (
                          <div className="route-card-delta">
                            <span>{distDelta >= 0 ? `+${distDelta.toFixed(1)}` : distDelta.toFixed(1)} km</span>
                            <span>{costDelta >= 0 ? `+${costDelta.toFixed(1)}` : costDelta.toFixed(1)} cost</span>
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
                <strong>Selected Route ({routeData.alternative_routes ? routeData.alternative_routes[selectedRouteIndex]?.route_name || "Optimal" : "Primary"})</strong>

                {(() => {
                  const currentR = routeData.alternative_routes
                    ? routeData.alternative_routes[selectedRouteIndex]
                    : routeData;
                  return (
                    <>
                      <div>Distance: {currentR.route.distance_km} km</div>
                      <div>Navigation cost: {currentR.route.total_navigation_cost}</div>
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

              <button
                className={`layer-button ${
                  displayMode === "icebergs" ? "active" : ""
                }`}
                onClick={() => setDisplayMode("icebergs")}
              >
                Icebergs
              </button>
            </div>

            {displayMode === "icebergs" && icebergData?.iceberg_coverage && (
              <div className="navigation-status iceberg-summary">
                <strong>Iceberg Coverage</strong>

                <div>
                  Represented: {icebergData.iceberg_coverage.represented_tracks}
                  /{icebergData.iceberg_coverage.total_tracks}
                </div>

                <div>
                  ML predictions: {icebergData.iceberg_coverage.ml_predictions}
                </div>

                <div>
                  Persistence:{" "}
                  {icebergData.iceberg_coverage.persistence_estimates}
                </div>

                <div>
                  Coverage: {icebergData.iceberg_coverage.coverage_percent}%
                </div>
              </div>
            )}
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

          {loadingIcebergs && (
            <div className="navigation-status">
              Loading iceberg predictions...
            </div>
          )}

          {icebergError && (
            <div className="navigation-status">
              Iceberg error: {icebergError}
            </div>
          )}

          {/* Dynamic navigation */}
          <section className="panel">
            <h2>Navigation</h2>

            <button
              className="secondary-button"
              onClick={handleSimulateProgress}
              disabled={!routeData}
            >
              🚢 Simulate Vessel Progress
            </button>

            <button
              className="replan-button"
              onClick={handleReplanRoute}
              disabled={!vesselPosition || loadingReplan}
            >
              {loadingReplan ? "Replanning..." : "🔄 Replan Route"}
            </button>

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
              routeIsReplanned={Boolean(replannedRouteData)}
              displayMode={displayMode}
              selectedRouteIndex={selectedRouteIndex}
              onSelectRoute={setSelectedRouteIndex}
            />
          </div>

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
    </div>
  );
}

export default App;
