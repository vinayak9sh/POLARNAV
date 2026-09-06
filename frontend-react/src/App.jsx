import "./App.css";
import AntarcticMap from "./components/AntarcticMap";
import { useEffect, useState } from "react";
import {
  getForecastGrid,
  getIcebergRiskGrid,
  calculateRoute,
  replanRoute,
} from "./services/api";

function App() {
  const [forecastDate, setForecastDate] = useState("2025-09-15");

  const [forecastData, setForecastData] = useState(null);

  const [loadingForecast, setLoadingForecast] = useState(false);

  const [forecastError, setForecastError] = useState(null);

  const [displayMode, setDisplayMode] = useState("sic");

  const [routeData, setRouteData] = useState(null);

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

    async function loadForecast() {
      setLoadingForecast(true);
      setForecastError(null);

      try {
        const data = await getForecastGrid(forecastDate);

        if (!cancelled) {
          setForecastData(data);
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
  }, [forecastDate]);

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

            <select
              id="forecastDate"
              value={forecastDate}
              onChange={(event) => setForecastDate(event.target.value)}
            >
              <option value="2025-09-15">15 Sep 2025</option>

              <option value="2025-09-20">20 Sep 2025</option>
            </select>

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

            {routeData && (
              <div className="navigation-status">
                <strong>Route calculated</strong>
                <br />
                {routeData.route.distance_km} km
                <br />
                Navigation cost: {routeData.route.total_navigation_cost}
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
                <strong>Initial Route</strong>

                <div>Distance: {routeData.route.distance_km} km</div>

                <div>Cost: {routeData.route.total_navigation_cost}</div>

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
                  </>
                )}
                {replannedRouteData && (
                  <>
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
                  const decision =
                    replannedRouteData?.decision || routeData?.decision;

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
                    ? "Initial route calculated. Simulate vessel progress to enable dynamic replanning."
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
                    ? "INITIAL ROUTE"
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
                    ? `${routeData.route.distance_km} km`
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
                    ? routeData.route.total_navigation_cost
                    : "—"}
              </strong>
            </div>

            {/* Decision factor */}
            <div>
              <span className="metric-label">Primary Factor</span>

              <strong>
                {replannedRouteData?.decision || routeData?.decision
                  ? (replannedRouteData?.decision || routeData?.decision)
                      .dominant_hazard === "sea_ice"
                    ? "Sea Ice"
                    : (replannedRouteData?.decision || routeData?.decision)
                          .dominant_hazard === "iceberg"
                      ? "Icebergs"
                      : "Combined"
                  : "—"}
              </strong>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
