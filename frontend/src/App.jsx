// ============================================================================
// frontend/src/App.jsx - Main Application Component
// ============================================================================

import React, { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

// Import custom components
import Header from './components/Header';
import LiveTelemetryTab from './components/LiveTelemetryTab';
import PredictionsTab from './components/PredictionsTab';
import AnalyticsTab from './components/AnalyticsTab';
import SystemTab from './components/SystemTab';
import { useWebSocket } from './hooks/useWebSocket';
import { useTelemetryData } from './hooks/useTelemetryData';

const App = () => {
  // State management
  const [activeTab, setActiveTab] = useState('live');
  
  // Custom hooks for data management
  const { connectionStatus, websocketManager } = useWebSocket();
  const { 
    telemetryData, 
    raceOutcome, 
    lapTimes, 
    driverPositions, 
    systemMetrics,
    updateDashboard 
  } = useTelemetryData();

  // Set up WebSocket message handler
  useEffect(() => {
    if (websocketManager) {
      websocketManager.setMessageHandler(updateDashboard);
    }
  }, [websocketManager, updateDashboard]);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <Header connectionStatus={connectionStatus} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 bg-gray-800">
            <TabsTrigger value="live" className="text-white data-[state=active]:bg-blue-600">
              Live Telemetry
            </TabsTrigger>
            <TabsTrigger value="predictions" className="text-white data-[state=active]:bg-green-600">
              Predictions
            </TabsTrigger>
            <TabsTrigger value="analytics" className="text-white data-[state=active]:bg-purple-600">
              Analytics
            </TabsTrigger>
            <TabsTrigger value="system" className="text-white data-[state=active]:bg-orange-600">
              System
            </TabsTrigger>
          </TabsList>

          <TabsContent value="live" className="mt-6">
            <LiveTelemetryTab 
              telemetryData={telemetryData}
              driverPositions={driverPositions}
            />
          </TabsContent>

          <TabsContent value="predictions" className="mt-6">
            <PredictionsTab 
              raceOutcome={raceOutcome}
              lapTimes={lapTimes}
            />
          </TabsContent>

          <TabsContent value="analytics" className="mt-6">
            <AnalyticsTab telemetryData={telemetryData} />
          </TabsContent>

          <TabsContent value="system" className="mt-6">
            <SystemTab systemMetrics={systemMetrics} />
          </TabsContent>
        </Tabs>

        {/* System Alerts */}
        {connectionStatus === 'error' && (
          <Alert className="mt-6 bg-red-900 border-red-700">
            <AlertDescription className="text-red-200">
              Connection to telemetry system lost. Attempting to reconnect...
            </AlertDescription>
          </Alert>
        )}

        {systemMetrics.gpu_utilization > 95 && (
          <Alert className="mt-6 bg-yellow-900 border-yellow-700">
            <AlertDescription className="text-yellow-200">
              High GPU utilization detected ({systemMetrics.gpu_utilization}%). 
              Consider scaling up GPU resources.
            </AlertDescription>
          </Alert>
        )}
      </main>
    </div>
  );
};

export default App;