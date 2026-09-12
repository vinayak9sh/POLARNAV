import React, { useMemo } from "react";

// Compass 16-point cardinal directions
function getCompassDirection(deg) {
  const directions = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
  ];
  const index = Math.round((((deg % 360) + 360) % 360) / 22.5) % 16;
  return directions[index];
}

// Great-circle bearing calculation in degrees (0..360)
function calculateBearing(lat1, lon1, lat2, lon2) {
  const phi1 = (lat1 * Math.PI) / 180;
  const phi2 = (lat2 * Math.PI) / 180;
  const deltaLon = ((lon2 - lon1) * Math.PI) / 180;

  const y = Math.sin(deltaLon) * Math.cos(phi2);
  const x =
    Math.cos(phi1) * Math.sin(phi2) -
    Math.sin(phi1) * Math.cos(phi2) * Math.cos(deltaLon);
  const theta = Math.atan2(y, x);
  return (((theta * 180) / Math.PI) + 360) % 360;
}

// Haversine distance in km
function calculateDistanceKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

// Normalize various coordinate formats to [{ lat, lon }]
function normalizeRoutePoints(routeCoordinates) {
  if (!routeCoordinates || !Array.isArray(routeCoordinates)) return [];
  return routeCoordinates
    .map((pt) => {
      const lon = Array.isArray(pt) ? Number(pt[0]) : Number(pt.longitude ?? pt.lon);
      const lat = Array.isArray(pt) ? Number(pt[1]) : Number(pt.latitude ?? pt.lat);
      return { lat, lon };
    })
    .filter((pt) => Number.isFinite(pt.lat) && Number.isFinite(pt.lon));
}

// Extract major navigation maneuvers / turns from polyline
function extractManeuvers(points) {
  if (points.length < 2) return { cumDists: [0], maneuvers: [] };

  const cumDists = [0];
  for (let i = 1; i < points.length; i++) {
    cumDists.push(
      cumDists[i - 1] +
        calculateDistanceKm(
          points[i - 1].lat,
          points[i - 1].lon,
          points[i].lat,
          points[i].lon
        )
    );
  }

  const maneuvers = [{ index: 0, point: points[0], cumDist: 0 }];
  let currentStartIdx = 0;
  let currentBearing = calculateBearing(
    points[0].lat,
    points[0].lon,
    points[1].lat,
    points[1].lon
  );

  for (let i = 1; i < points.length - 1; i++) {
    const segBearing = calculateBearing(
      points[i].lat,
      points[i].lon,
      points[i + 1].lat,
      points[i + 1].lon
    );
    let diff = ((segBearing - currentBearing + 540) % 360) - 180;
    const distFromStart = cumDists[i] - cumDists[currentStartIdx];

    // Detect significant turn (>= 12 deg) after traveling at least 3 km
    if (Math.abs(diff) >= 12 && distFromStart >= 3.0) {
      maneuvers.push({ index: i, point: points[i], cumDist: cumDists[i] });
      currentStartIdx = i;
      currentBearing = segBearing;
    }
  }

  // Ensure destination is the last maneuver
  const lastIdx = points.length - 1;
  if (maneuvers[maneuvers.length - 1].index !== lastIdx) {
    maneuvers.push({
      index: lastIdx,
      point: points[lastIdx],
      cumDist: cumDists[lastIdx],
    });
  }

  return { cumDists, maneuvers };
}

function NavigationGuidanceHUD({
  routeData,
  replannedRouteData,
  vesselPosition,
  vesselHeading = 0,
  vesselSpeed = 0,
  selectedRouteIndex = 0,
}) {
  const baseRoute = replannedRouteData || routeData;
  const isReplanned = Boolean(replannedRouteData);

  const activeRoute = useMemo(() => {
    if (!baseRoute) return null;
    if (
      Array.isArray(baseRoute.alternative_routes) &&
      baseRoute.alternative_routes.length > 0
    ) {
      const idx =
        selectedRouteIndex >= 0 &&
        selectedRouteIndex < baseRoute.alternative_routes.length
          ? selectedRouteIndex
          : 0;
      return baseRoute.alternative_routes[idx];
    }
    return baseRoute;
  }, [baseRoute, selectedRouteIndex]);

  const guidance = useMemo(() => {
    const rawCoords =
      activeRoute?.geometry?.coordinates || activeRoute?.coordinates;
    const points = normalizeRoutePoints(rawCoords);

    if (points.length < 2) {
      return null;
    }

    const { cumDists, maneuvers } = extractManeuvers(points);
    const totalRouteDistKm = cumDists[cumDists.length - 1];

    const currentLat = vesselPosition
      ? Number(vesselPosition.latitude)
      : points[0].lat;
    const currentLon = vesselPosition
      ? Number(vesselPosition.longitude)
      : points[0].lon;

    if (!Number.isFinite(currentLat) || !Number.isFinite(currentLon)) {
      return null;
    }

    // Project vessel position onto polyline to find progress along the route
    let minPerpDist = Infinity;
    let bestSegmentIdx = 0;
    let bestT = 0;

    for (let i = 0; i < points.length - 1; i++) {
      const a = points[i];
      const b = points[i + 1];

      const cosLat = Math.cos((a.lat * Math.PI) / 180.0);
      const dxV = (currentLon - a.lon) * 111.0 * cosLat;
      const dyV = (currentLat - a.lat) * 111.0;

      const dxB = (b.lon - a.lon) * 111.0 * cosLat;
      const dyB = (b.lat - a.lat) * 111.0;

      const lenSq = dxB * dxB + dyB * dyB;

      let t = 0;
      let dist = 0;

      if (lenSq === 0) {
        dist = Math.hypot(dxV, dyV);
      } else {
        t = Math.max(0, Math.min(1, (dxV * dxB + dyV * dyB) / lenSq));
        const qx = t * dxB;
        const qy = t * dyB;
        dist = Math.hypot(dxV - qx, dyV - qy);
      }

      if (dist < minPerpDist) {
        minPerpDist = dist;
        bestSegmentIdx = i;
        bestT = t;
      }
    }

    // Cumulative distance traversed along route
    const vesselRouteDist =
      cumDists[bestSegmentIdx] +
      bestT * (cumDists[bestSegmentIdx + 1] - cumDists[bestSegmentIdx]);

    // Find the upcoming maneuver waypoint (must be ahead of vessel by >= 0.3 km)
    let nextManeuverIdx = 1;
    for (let m = 1; m < maneuvers.length; m++) {
      if (maneuvers[m].cumDist > vesselRouteDist + 0.3) {
        nextManeuverIdx = m;
        break;
      }
      nextManeuverIdx = m;
    }

    const nextManeuver = maneuvers[nextManeuverIdx];
    const targetPoint = nextManeuver.point;

    // Remaining distance along route to next maneuver waypoint
    const distanceToNextKm = Math.max(
      0.1,
      nextManeuver.cumDist - vesselRouteDist
    );

    // Total remaining distance to destination
    const totalRemainingKm = Math.max(0.1, totalRouteDistKm - vesselRouteDist);

    // Bearing to steer along the active corridor / next point
    // We look ahead slightly on polyline for the local track bearing
    const lookaheadIdx = Math.min(
      points.length - 1,
      bestSegmentIdx + (bestT > 0.8 ? 2 : 1)
    );
    const lookaheadPoint = points[lookaheadIdx];

    const targetBearing = calculateBearing(
      currentLat,
      currentLon,
      lookaheadPoint.lat,
      lookaheadPoint.lon
    );
    const roundedBearing = Math.round(targetBearing);
    const cardinalDir = getCompassDirection(roundedBearing);
    const bearingLabel = `${String(roundedBearing).padStart(3, "0")}° ${cardinalDir}`;

    // Turn instruction at the upcoming maneuver waypoint
    let turnType = "straight";
    let turnDegrees = 0;
    let turnInstruction = "then continue straight";
    let turnIcon = "↟";

    const isDestination = nextManeuverIdx === maneuvers.length - 1;

    if (!isDestination && nextManeuverIdx < maneuvers.length - 1) {
      const currentLegManeuver = maneuvers[nextManeuverIdx - 1];
      const nextLegManeuver = maneuvers[nextManeuverIdx + 1];

      const bearingIn = calculateBearing(
        currentLegManeuver.point.lat,
        currentLegManeuver.point.lon,
        nextManeuver.point.lat,
        nextManeuver.point.lon
      );
      const bearingOut = calculateBearing(
        nextManeuver.point.lat,
        nextManeuver.point.lon,
        nextLegManeuver.point.lat,
        nextLegManeuver.point.lon
      );

      let deltaTurn = ((bearingOut - bearingIn + 540) % 360) - 180;
      const absTurn = Math.round(Math.abs(deltaTurn));

      if (deltaTurn >= 8) {
        turnType = "right";
        turnDegrees = absTurn;
        turnInstruction = `then turn ${absTurn}° right`;
        turnIcon = "↱";
      } else if (deltaTurn <= -8) {
        turnType = "left";
        turnDegrees = absTurn;
        turnInstruction = `then turn ${absTurn}° left`;
        turnIcon = "↰";
      } else {
        turnType = "straight";
        turnInstruction = "then continue straight";
        turnIcon = "↟";
      }
    } else {
      turnType = "arrival";
      turnInstruction = "then arrive at destination";
      turnIcon = "🏁";
    }

    // Live Steer alignment advisory
    const normalizedHeading = ((vesselHeading % 360) + 360) % 360;
    const headingDiff = ((targetBearing - normalizedHeading + 540) % 360) - 180;
    const absSteer = Math.round(Math.abs(headingDiff));

    let steerAdvice = "✅ On Course";
    let steerClass = "on-course";
    if (headingDiff > 5) {
      steerAdvice = `Steer ${absSteer}° Right (↻)`;
      steerClass = "steer-right";
    } else if (headingDiff < -5) {
      steerAdvice = `Steer ${absSteer}° Left (↺)`;
      steerClass = "steer-left";
    }

    // ETA calculation
    let etaText = "—";
    if (vesselSpeed > 0) {
      const speedKmH = vesselSpeed * 1.852;
      const etaMin = Math.max(1, Math.round((distanceToNextKm / speedKmH) * 60));
      if (etaMin >= 60) {
        const hours = Math.floor(etaMin / 60);
        const mins = etaMin % 60;
        etaText = `${hours}h ${mins}m`;
      } else {
        etaText = `${etaMin} min`;
      }
    } else {
      etaText = "Stopped (0 kts)";
    }

    const routeLabel = isReplanned
      ? "Replanned Route"
      : activeRoute.label || activeRoute.route_name || "Primary Route";

    return {
      routeLabel,
      isReplanned,
      bearingLabel,
      targetBearing: roundedBearing,
      distanceToNextKm: distanceToNextKm.toFixed(1),
      turnType,
      turnDegrees,
      turnInstruction,
      turnIcon,
      steerAdvice,
      steerClass,
      totalRemainingKm: totalRemainingKm.toFixed(1),
      etaText,
      targetWaypointIndex: nextManeuverIdx,
      totalWaypoints: Math.max(1, maneuvers.length - 1),
    };
  }, [activeRoute, isReplanned, vesselPosition, vesselHeading, vesselSpeed]);

  if (!guidance) {
    return null;
  }

  return (
    <aside
      className={`nav-guidance-hud ${guidance.isReplanned ? "replanned" : ""}`}
      aria-label="Navigation Guidance HUD"
    >
      {/* Top HUD Header */}
      <div className="nav-hud-header">
        <div className="nav-hud-title-wrap">
          <span className="nav-hud-badge">NAV GUIDANCE</span>
          <span className={`nav-route-tag ${guidance.isReplanned ? "replanned-tag" : ""}`}>
            {guidance.routeLabel}
          </span>
        </div>
        <div className="nav-hud-waypoint-counter">
          Leg {guidance.targetWaypointIndex}/{guidance.totalWaypoints}
        </div>
      </div>

      {/* Main Guidance Banner */}
      <div className="nav-main-instruction-box">
        <div className={`nav-turn-icon-box turn-${guidance.turnType}`}>
          <span className="nav-turn-glyph">{guidance.turnIcon}</span>
        </div>
        <div className="nav-instruction-text-block">
          <div className="nav-primary-instruction">
            Head <span className="highlight-bearing">{guidance.bearingLabel}</span> for next{" "}
            <span className="highlight-distance">{guidance.distanceToNextKm} km</span>
          </div>
          <div className="nav-secondary-instruction">
            {guidance.turnInstruction}
          </div>
        </div>
      </div>

      {/* Live Telemetry & Steer Advisory Grid */}
      <div className="nav-hud-telemetry-grid">
        <div className="nav-telemetry-cell">
          <span className="telemetry-label">STEER ADVICE</span>
          <span className={`telemetry-val ${guidance.steerClass}`}>
            {guidance.steerAdvice}
          </span>
        </div>

        <div className="nav-telemetry-cell">
          <span className="telemetry-label">LEG ETA</span>
          <span className="telemetry-val">{guidance.etaText}</span>
        </div>

        <div className="nav-telemetry-cell">
          <span className="telemetry-label">TOTAL REMAINING</span>
          <span className="telemetry-val">{guidance.totalRemainingKm} km</span>
        </div>
      </div>
    </aside>
  );
}

export default NavigationGuidanceHUD;
