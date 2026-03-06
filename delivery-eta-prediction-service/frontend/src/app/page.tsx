"use client";

import React, { useState, useRef, useMemo } from 'react';
import Map, { Marker, Source, Layer } from 'react-map-gl';
import { Clock, User, Plus, Search, MapPin, Navigation, Tag } from 'lucide-react';
import * as turf from '@turf/turf';

// Import the dynamically generated list of all 263 NYC Taxi Zones
import NYC_ZONES from './nyc_zones.json';

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

export default function Home() {
    const mapRef = useRef<any>(null);
    const [viewState, setViewState] = useState({ longitude: -73.98, latitude: 40.75, zoom: 11 });
    const [pickupId, setPickupId] = useState<string>("");
    const [dropoffId, setDropoffId] = useState<string>("");
    const [prediction, setPrediction] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [routeGeoJSON, setRouteGeoJSON] = useState<any>(null);

    const pickupZone = useMemo(() => NYC_ZONES.find(z => z.id === parseInt(pickupId)), [pickupId]);
    const dropoffZone = useMemo(() => NYC_ZONES.find(z => z.id === parseInt(dropoffId)), [dropoffId]);

    const handlePredict = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!pickupZone || !dropoffZone) return;

        setLoading(true);
        setPrediction(null);
        setRouteGeoJSON(null);

        try {
            const res = await fetch("http://localhost:8000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    pickup_location_id: pickupZone.id,
                    dropoff_location_id: dropoffZone.id,
                }),
            });
            const data = await res.json();
            setPrediction(data);

            const mapboxReq = await fetch(
                `https://api.mapbox.com/directions/v5/mapbox/driving-traffic/${pickupZone.lng},${pickupZone.lat};${dropoffZone.lng},${dropoffZone.lat}?geometries=geojson&access_token=${MAPBOX_TOKEN}`
            );
            const mapboxData = await mapboxReq.json();

            if (mapboxData.routes?.length > 0) {
                const geo = mapboxData.routes[0].geometry;
                setRouteGeoJSON({ type: "Feature", properties: {}, geometry: geo });
                if (mapRef.current) {
                    const bbox = turf.bbox(turf.lineString(geo.coordinates));
                    mapRef.current.fitBounds([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], { padding: 80, duration: 1500 });
                }
            }
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ position: "relative", width: "100vw", height: "100vh", overflow: "hidden", background: "#000" }}>

            {/* ── Map ── */}
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, zIndex: 0 }}>
                <Map
                    ref={mapRef}
                    {...viewState}
                    onMove={evt => setViewState(evt.viewState)}
                    mapStyle="mapbox://styles/mapbox/dark-v11"
                    mapboxAccessToken={MAPBOX_TOKEN}
                    style={{ width: "100%", height: "100%" }}
                >
                    {routeGeoJSON && (
                        <Source id="route" type="geojson" data={routeGeoJSON}>
                            <Layer
                                id="route-line" type="line"
                                layout={{ "line-join": "round", "line-cap": "round" }}
                                paint={{ "line-color": "#f59e0b", "line-width": 4 }}
                            />
                        </Source>
                    )}
                    {pickupZone && (
                        <Marker longitude={pickupZone.lng} latitude={pickupZone.lat} anchor="bottom">
                            <div style={{ width: 14, height: 14, borderRadius: "50%", background: "#f59e0b", border: "2.5px solid #fff", boxShadow: "0 0 0 3px rgba(245,158,11,0.25)" }} />
                        </Marker>
                    )}
                    {dropoffZone && (
                        <Marker longitude={dropoffZone.lng} latitude={dropoffZone.lat} anchor="bottom">
                            <div style={{ width: 14, height: 14, background: "#18181b", border: "2.5px solid #fff", boxShadow: "0 2px 8px rgba(0,0,0,0.3)" }} />
                        </Marker>
                    )}
                </Map>
            </div>

            {/* ── Form Panel ── */}
            <div style={{
                position: "absolute", top: 32, left: 32, zIndex: 50,
                width: 380,
                background: "#0f0f11",
                border: "1px solid rgba(255,255,255,0.07)",
                borderRadius: 20,
                boxShadow: "0 32px 64px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.03)",
                overflow: "hidden",
                fontFamily: "var(--font-geist-sans), sans-serif",
            }}>

                {/* Header */}
                <div style={{ padding: "24px 24px 0" }}>
                    <div style={{
                        display: "flex", alignItems: "center", gap: 7,
                        marginBottom: 12,
                    }}>
                        <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#f59e0b", boxShadow: "0 0 6px #f59e0b" }} />
                        <span style={{ fontFamily: "var(--font-geist-mono), monospace", fontSize: 10, color: "#71717a", letterSpacing: "2px", textTransform: "uppercase" }}>
                            PHILIP'S ML POWERED
                        </span>
                    </div>
                    <h1 style={{ fontSize: 16, fontWeight: 700, color: "#fafafa", letterSpacing: "-0.4px", lineHeight: 1.15 }}>
                        Trip Duration and Cost Estimator
                    </h1>
                </div>

                {/* Divider */}
                <div style={{ height: 1, background: "rgba(255,255,255,0.06)", margin: "20px 0 0" }} />

                {/* Form Body */}
                <form onSubmit={handlePredict} style={{ padding: "20px 24px 24px" }}>

                    {/* Route inputs */}
                    <div style={{ position: "relative", marginBottom: 16 }}>

                        {/* connector line */}
                        <div style={{
                            position: "absolute", left: 19, top: 38, bottom: 38,
                            width: 1,
                            background: "linear-gradient(to bottom, #f59e0b55, #ffffff22)",
                        }} />

                        {/* Pickup */}
                        <div style={{
                            display: "flex", alignItems: "center", gap: 12,
                            background: "#18181b", border: "1px solid rgba(255,255,255,0.07)",
                            borderRadius: 12, padding: "11px 14px",
                            marginBottom: 6, transition: "border-color 0.2s",
                        }}>
                            <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#f59e0b", boxShadow: "0 0 5px #f59e0b", flexShrink: 0 }} />
                            <div style={{ flex: 1 }}>
                                <div style={{ fontFamily: "var(--font-geist-mono)", fontSize: 9, color: "#52525b", textTransform: "uppercase", letterSpacing: "1.5px", marginBottom: 3 }}>
                                    From
                                </div>
                                <select
                                    value={pickupId}
                                    onChange={e => setPickupId(e.target.value)}
                                    required
                                    style={{
                                        width: "100%", background: "none", border: "none", outline: "none",
                                        color: pickupId ? "#f4f4f5" : "#52525b",
                                        fontSize: 14, fontWeight: 500,
                                        fontFamily: "var(--font-geist-sans)",
                                        cursor: "pointer", appearance: "none",
                                    }}
                                >
                                    <option value="" disabled style={{ background: "#18181b", color: "#71717a" }}>Select pickup</option>
                                    {NYC_ZONES.map(z => <option key={`p-${z.id}`} value={z.id} style={{ background: "#18181b", color: "#f4f4f5" }}>{z.name}</option>)}
                                </select>
                            </div>
                        </div>

                        {/* Dropoff */}
                        <div style={{
                            display: "flex", alignItems: "center", gap: 12,
                            background: "#18181b", border: "1px solid rgba(255,255,255,0.07)",
                            borderRadius: 12, padding: "11px 14px",
                            transition: "border-color 0.2s",
                        }}>
                            <div style={{ width: 8, height: 8, background: "#f4f4f5", flexShrink: 0, boxShadow: "0 0 4px rgba(255,255,255,0.3)" }} />
                            <div style={{ flex: 1 }}>
                                <div style={{ fontFamily: "var(--font-geist-mono)", fontSize: 9, color: "#52525b", textTransform: "uppercase", letterSpacing: "1.5px", marginBottom: 3 }}>
                                    To
                                </div>
                                <select
                                    value={dropoffId}
                                    onChange={e => setDropoffId(e.target.value)}
                                    required
                                    style={{
                                        width: "100%", background: "none", border: "none", outline: "none",
                                        color: dropoffId ? "#f4f4f5" : "#52525b",
                                        fontSize: 14, fontWeight: 500,
                                        fontFamily: "var(--font-geist-sans)",
                                        cursor: "pointer", appearance: "none",
                                    }}
                                >
                                    <option value="" disabled style={{ background: "#18181b", color: "#71717a" }}>Select destination</option>
                                    {NYC_ZONES.map(z => <option key={`d-${z.id}`} value={z.id} style={{ background: "#18181b", color: "#f4f4f5" }}>{z.name}</option>)}
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* CTA */}
                    <button
                        type="submit"
                        disabled={loading || !pickupId || !dropoffId}
                        style={{
                            width: "100%", padding: "13px 0",
                            background: loading || !pickupId || !dropoffId ? "#27272a" : "#f59e0b",
                            color: loading || !pickupId || !dropoffId ? "#52525b" : "#000",
                            border: "none", borderRadius: 12,
                            fontSize: 13, fontWeight: 700,
                            fontFamily: "var(--font-geist-sans)",
                            letterSpacing: "0.2px",
                            cursor: loading || !pickupId || !dropoffId ? "not-allowed" : "pointer",
                            display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                            transition: "background 0.2s, color 0.2s, box-shadow 0.2s",
                            boxShadow: loading || !pickupId || !dropoffId ? "none" : "0 4px 20px rgba(245,158,11,0.25)",
                        }}
                    >
                        {loading ? (
                            <>
                                <span style={{
                                    width: 13, height: 13, border: "2px solid rgba(255,255,255,0.2)",
                                    borderTopColor: "#71717a", borderRadius: "50%",
                                    display: "inline-block",
                                    animation: "spin 0.6s linear infinite",
                                }} />
                                Calculating...
                            </>
                        ) : (
                            <>
                                Predict Trip
                                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M3 8h10M9 4l4 4-4 4" />
                                </svg>
                            </>
                        )}
                    </button>
                </form>

                {/* ── Results ── */}
                {prediction && (
                    <>
                        <div style={{ height: 1, background: "rgba(255,255,255,0.06)" }} />
                        <div style={{ padding: "20px 24px 24px" }}>

                            <div style={{ fontFamily: "var(--font-geist-mono)", fontSize: 9, color: "#52525b", textTransform: "uppercase", letterSpacing: "2px", marginBottom: 14 }}>
                                Estimate
                            </div>

                            {/* Two stat tiles */}
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>

                                {/* Minutes */}
                                <div style={{
                                    background: "rgba(245,158,11,0.07)",
                                    border: "1px solid rgba(245,158,11,0.15)",
                                    borderRadius: 12, padding: "16px 14px",
                                }}>
                                    <div style={{ fontFamily: "var(--font-geist-mono)", fontSize: 9, color: "#92400e", textTransform: "uppercase", letterSpacing: "1.5px", marginBottom: 8 }}>
                                        Minutes
                                    </div>
                                    <div style={{ fontSize: 30, fontWeight: 700, color: "#f59e0b", letterSpacing: "-1.5px", lineHeight: 1, fontFamily: "var(--font-geist-mono)" }}>
                                        {prediction.estimated_time_minutes}
                                    </div>
                                    <div style={{ fontSize: 10, color: "#71717a", marginTop: 4, fontFamily: "var(--font-geist-mono)", textTransform: "uppercase", letterSpacing: "1px" }}>
                                        delivery time
                                    </div>
                                </div>

                                {/* Cost */}
                                <div style={{
                                    background: "rgba(16,185,129,0.07)",
                                    border: "1px solid rgba(16,185,129,0.15)",
                                    borderRadius: 12, padding: "16px 14px",
                                }}>
                                    <div style={{ fontFamily: "var(--font-geist-mono)", fontSize: 9, color: "#065f46", textTransform: "uppercase", letterSpacing: "1.5px", marginBottom: 8 }}>
                                        Est. Cost
                                    </div>
                                    <div style={{ fontSize: 30, fontWeight: 700, color: "#10b981", letterSpacing: "-1.5px", lineHeight: 1, fontFamily: "var(--font-geist-mono)" }}>
                                        ${prediction.estimated_cost_usd}
                                    </div>
                                    <div style={{ fontSize: 10, color: "#71717a", marginTop: 4, fontFamily: "var(--font-geist-mono)", textTransform: "uppercase", letterSpacing: "1px" }}>
                                        USD
                                    </div>
                                </div>
                            </div>

                            {/* Distance footer */}
                            <div style={{
                                display: "flex", alignItems: "center", justifyContent: "space-between",
                                background: "#18181b", border: "1px solid rgba(255,255,255,0.06)",
                                borderRadius: 10, padding: "10px 14px",
                            }}>
                                <span style={{ fontFamily: "var(--font-geist-mono)", fontSize: 10, color: "#52525b", textTransform: "uppercase", letterSpacing: "1px" }}>
                                    Route distance
                                </span>
                                <span style={{ fontFamily: "var(--font-geist-mono)", fontSize: 13, fontWeight: 500, color: "#f4f4f5" }}>
                                    {prediction.historical_route_stats?.avg_distance_miles} mi
                                </span>
                            </div>
                        </div>
                    </>
                )}
            </div>

            <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        select option { background: #18181b; color: #f4f4f5; }
      `}</style>
        </div>
    );
}