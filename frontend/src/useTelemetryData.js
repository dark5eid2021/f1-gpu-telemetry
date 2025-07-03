// ============================================================================
// frontend/src/hooks/useTelemetryData.js - Telemetry Data Management Hook
// ============================================================================

import { useState, useEffect, useCallback } from 'react';

export const useTelemetryData = () => {
  const [telemetryData, setTelemetryData] = useState([]);
  const [raceOutcome, setRaceOutcome] = useState(null);
  const [lapTimes, setLapTimes] = useState([]);
  const [driverPositions, setDriverPositions] = useState([]);
  const [systemMetrics, setSystemMetrics] = useState({});

  // Update dashboard with new data from WebSocket
  const updateDashboard = useCallback((data) => {
    if (data.race_outcome) {
      setRaceOutcome(data.race_outcome);
    }
    
    if (data.telemetry) {
      setTelemetryData(prev => {
        const newData = [...prev, ...data.telemetry].slice(-1000); // Keep last 1000 points
        return newData;
      });
    }
    
    if (data.lap_times) {
      setLapTimes(data.lap_times);
    }
    
    if (data.positions) {
      setDriverPositions(data.positions);
    }
    
    if (data.system_metrics) {
      setSystemMetrics(data.system_metrics);
    }
  }, []);

  // Fetch initial data on component mount
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [predictionsRes, telemetryRes, healthRes] = await Promise.all([
          fetch('/api/v1/predictions/race'),
          fetch('/api/v1/telemetry/live'),
          fetch('/api/v1/health')
        ]);
        
        if (predictionsRes.ok) {
          const predictions = await predictionsRes.json();
          setRaceOutcome(predictions.race_outcome);
        }
        
        if (telemetryRes.ok) {
          const telemetry = await telemetryRes.json();
          setTelemetryData(telemetry.data || []);
        }
        
        if (healthRes.ok) {
          const health = await healthRes.json();
          setSystemMetrics(health);
        }
      } catch (error) {
        console.error('Error fetching initial data:', error);
      }
    };

    fetchInitialData();
  }, []);

  return {
    telemetryData,
    raceOutcome,
    lapTimes,
    driverPositions,
    systemMetrics,
    updateDashboard
  };
};