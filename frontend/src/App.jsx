// ============================================================================
// React Dashboard - Real-time F1 Telemetry Visualization
// File: frontend/src/App.jsx
// ============================================================================

import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, AreaChart, Area, BarChart, Bar, ScatterChart,
  Scatter, Cell, PieChart, Pie
} from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';

const App = () => {
  // State management
  const [telemetryData, setTelemetryData] = useState([]);
  const [raceOutcome, setRaceOutcome] = useState(null);
  const [lapTimes, setLapTimes] = useState([]);
  const [driverPositions, setDriverPositions] = useState([]);
  const [systemMetrics, setSystemMetrics] = useState({});
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [activeTab, setActiveTab] = useState('live');
  
  const wsRef = useRef(null);
  const canvasRef = useRef(null);

  // WebSocket connection for real-time data
  useEffect(() => {
    const connectWebSocket = () => {
      const wsUrl = process.env.NODE_ENV === 'production' 
        ? 'wss://f1-gpu.yourdomain.com/ws/telemetry'
        : 'ws://localhost:8000/ws/telemetry';
      
      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        setConnectionStatus('connected');
        console.log('WebSocket connected');
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          updateDashboard(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      wsRef.current.onclose = () => {
        setConnectionStatus('disconnected');
        console.log('WebSocket disconnected, attempting to reconnect...');
        setTimeout(connectWebSocket, 3000);
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      };
    };

    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Update dashboard with new data
  const updateDashboard = (data) => {
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
  };

  // Fetch initial data
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

  // Driver colors for consistent visualization
  const driverColors = {
    1: '#FF6B6B', 44: '#4ECDC4', 16: '#45B7D1', 55: '#96CEB4',
    4: '#FFEAA7', 77: '#DDA0DD', 11: '#98D8C8', 31: '#F7DC6F',
    63: '#BB8FCE', 18: '#85C1E9', 5: '#F8C471', 47: '#82E0AA',
    24: '#F1948A', 81: '#AED6F1', 20: '#A9DFBF', 2: '#F9E79F',
    10: '#D7DBDD', 27: '#FADBD8', 22: '#E8DAEF', 23: '#D6EAF8'
  };

  // Connection status indicator
  const ConnectionStatus = () => (
    <div className="flex items-center space-x-2">
      <div className={`w-3 h-3 rounded-full ${
        connectionStatus === 'connected' ? 'bg-green-500' : 
        connectionStatus === 'error' ? 'bg-red-500' : 'bg-yellow-500'
      }`} />
      <span className="text-sm font-medium">
        {connectionStatus === 'connected' ? 'Live' : 
         connectionStatus === 'error' ? 'Error' : 'Connecting...'}
      </span>
    </div>
  );

  // Live telemetry chart component
  const TelemetryChart = ({ data, selectedMetric = 'speed' }) => {
    const chartData = data.slice(-100).map((point, index) => ({
      time: index,
      value: point[selectedMetric] || 0,
      driver: point.driver_id
    }));

    return (
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip 
            formatter={(value, name) => [value, selectedMetric]}
            labelFormatter={(time) => `Time: ${time}s`}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="value" 
            stroke="#8884d8" 
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    );
  };

  // Race outcome prediction component
  const RaceOutcomePrediction = ({ predictions }) => {
    if (!predictions || predictions.length === 0) {
      return <div className="text-center text-gray-500">No predictions available</div>;
    }

    const chartData = predictions.map((prob, index) => ({
      position: index + 1,
      probability: prob * 100,
      driver: index + 1
    }));

    return (
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="position" />
          <YAxis />
          <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Win Probability']} />
          <Bar dataKey="probability">
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={driverColors[entry.driver] || '#8884d8'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  };

  // Track position visualization
  const TrackPositionMap = () => {
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas || driverPositions.length === 0) return;

      const ctx = canvas.getContext('2d');
      const width = canvas.width = canvas.offsetWidth;
      const height = canvas.height = canvas.offsetHeight;

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Draw track outline (simplified)
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.ellipse(width/2, height/2, width*0.4, height*0.3, 0, 0, 2 * Math.PI);
      ctx.stroke();

      // Draw cars
      driverPositions.forEach((driver, index) => {
        const angle = (driver.position / 100) * 2 * Math.PI;
        const x = width/2 + Math.cos(angle) * width * 0.35;
        const y = height/2 + Math.sin(angle) * height * 0.25;

        ctx.fillStyle = driverColors[driver.number] || '#8884d8';
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fill();

        // Driver number
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(driver.number, x, y + 4);
      });
    }, [driverPositions]);

    return (
      <canvas 
        ref={canvasRef} 
        className="w-full h-64 border rounded-lg"
        style={{ maxWidth: '100%', height: '256px' }}
      />
    );
  };

  // System metrics component
  const SystemMetrics = ({ metrics }) => (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <Card>
        <CardContent className="p-4">
          <div className="text-2xl font-bold text-green-600">
            {metrics.gpu_utilization || 0}%
          </div>
          <p className="text-sm text-gray-600">GPU Utilization</p>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="p-4">
          <div className="text-2xl font-bold text-blue-600">
            {metrics.processing_latency || 0}ms
          </div>
          <p className="text-sm text-gray-600">Processing Latency</p>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="p-4">
          <div className="text-2xl font-bold text-purple-600">
            {metrics.throughput || 0}
          </div>
          <p className="text-sm text-gray-600">Events/sec</p>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="p-4">
          <div className="text-2xl font-bold text-orange-600">
            {metrics.memory_usage || 0}%
          </div>
          <p className="text-sm text-gray-600">Memory Usage</p>
        </CardContent>
      </Card>
    </div>
  );

  // Lap times comparison
  const LapTimesChart = ({ data }) => {
    if (!data || data.length === 0) return null;

    return (
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="lap" />
          <YAxis />
          <Tooltip />
          <Legend />
          {Object.keys(driverColors).slice(0, 5).map(driver => (
            <Line 
              key={driver}
              type="monotone" 
              dataKey={`driver_${driver}`}
              stroke={driverColors[driver]}
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    );
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <h1 className="text-3xl font-bold text-white">
                🏎️ F1 GPU Telemetry System
              </h1>
              <Badge variant="secondary" className="ml-4">
                Real-time Analytics
              </Badge>
            </div>
            <div className="flex items-center space-x-4">
              <ConnectionStatus />
              <Button 
                variant="outline" 
                onClick={() => window.location.reload()}
                className="text-white border-white hover:bg-white hover:text-gray-900"
              >
                Refresh
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
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

          {/* Live Telemetry Tab */}
          <TabsContent value="live" className="mt-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Real-time Speed Chart */}
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Live Speed Data</CardTitle>
                </CardHeader>
                <CardContent>
                  <TelemetryChart data={telemetryData} selectedMetric="speed" />
                </CardContent>
              </Card>

              {/* Track Position Map */}
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Track Positions</CardTitle>
                </CardHeader>
                <CardContent>
                  <TrackPositionMap />
                </CardContent>
              </Card>

              {/* Throttle & Brake Data */}
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Throttle & Brake</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={telemetryData.slice(-50)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Area 
                        type="monotone" 
                        dataKey="throttle" 
                        stackId="1"
                        stroke="#82ca9d" 
                        fill="#82ca9d" 
                      />
                      <Area 
                        type="monotone" 
                        dataKey="brake" 
                        stackId="2"
                        stroke="#ff7300" 
                        fill="#ff7300" 
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Tire Temperature */}
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Tire Temperatures</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={telemetryData.slice(-30)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="tire_temp_fl" stroke="#ff6b6b" name="Front Left" />
                      <Line type="monotone" dataKey="tire_temp_fr" stroke="#4ecdc4" name="Front Right" />
                      <Line type="monotone" dataKey="tire_temp_rl" stroke="#45b7d1" name="Rear Left" />
                      <Line type="monotone" dataKey="tire_temp_rr" stroke="#96ceb4" name="Rear Right" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Predictions Tab */}
          <TabsContent value="predictions" className="mt-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Race Outcome Prediction */}
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Race Outcome Prediction</CardTitle>
                  <p className="text-gray-400">AI-powered finishing position probabilities</p>
                </CardHeader>
                <CardContent>
                  <RaceOutcomePrediction predictions={raceOutcome} />
                </CardContent>
              </Card>

              {/* Lap Time Predictions */}
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Lap Time Predictions</CardTitle>
                  <p className="text-gray-400">Next lap time estimates</p>
                </CardHeader>
                <CardContent>
                  <LapTimesChart data={lapTimes} />
                </CardContent>
              </Card>

              {/* Pit Strategy Recommendations */}
              <Card className="bg-gray-800 border-gray-700 lg:col-span-2">
                <CardHeader>
                  <CardTitle className="text-white">Pit Strategy Recommendations</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {[1, 44, 16].map(driver => (
                      <div key={driver} className="p-4 bg-gray-700 rounded-lg">
                        <div className="flex items-center mb-2">
                          <div 
                            className="w-4 h-4 rounded-full mr-2"
                            style={{ backgroundColor: driverColors[driver] }}
                          />
                          <span className="font-semibold">Driver {driver}</span>
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span>Optimal Pit:</span>
                            <span className="text-green-400">Lap {15 + driver}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Tire Strategy:</span>
                            <span className="text-blue-400">Medium → Soft</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Time Loss:</span>
                            <span className="text-yellow-400">22.{driver}s</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="mt-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Performance Metrics */}
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Performance Analytics</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart data={telemetryData.slice(-100)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="speed" name="Speed" />
                      <YAxis dataKey="throttle" name="Throttle" />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter name="Speed vs Throttle" data={telemetryData.slice(-100)} fill="#8884d8" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Sector Times */}
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white">Sector Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {['Sector 1', 'Sector 2', 'Sector 3'].map((sector, index) => (
                      <div key={sector}>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium">{sector}</span>
                          <span className="text-sm">
                            {(25.123 + index * 0.5).toFixed(3)}s
                          </span>
                        </div>
                        <Progress 
                          value={75 + index * 5} 
                          className="h-2"
                        />
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Driver Comparison */}
              <Card className="bg-gray-800 border-gray-700 lg:col-span-2">
                <CardHeader>
                  <CardTitle className="text-white">Driver Performance Comparison</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={[
                      { name: 'Driver 1', avgSpeed: 315, consistency: 92, skill: 88 },
                      { name: 'Driver 44', avgSpeed: 312, consistency: 89, skill: 91 },
                      { name: 'Driver 16', avgSpeed: 318, consistency: 87, skill: 85 },
                      { name: 'Driver 55', avgSpeed: 310, consistency: 94, skill: 89 },
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="avgSpeed" fill="#8884d8" name="Avg Speed (km/h)" />
                      <Bar dataKey="consistency" fill="#82ca9d" name="Consistency %" />
                      <Bar dataKey="skill" fill="#ffc658" name="Skill Rating" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* System Tab */}
          <TabsContent value="system" className="mt-6">
            <div className="space-y-6">
              {/* System Metrics */}
              <SystemMetrics metrics={systemMetrics} />

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* GPU Performance */}
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle className="text-white">GPU Performance</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={[
                        { time: '10:00', utilization: 85, memory: 78, temperature: 72 },
                        { time: '10:05', utilization: 92, memory: 82, temperature: 75 },
                        { time: '10:10', utilization: 88, memory: 79, temperature: 73 },
                        { time: '10:15', utilization: 94, memory: 85, temperature: 77 },
                        { time: '10:20', utilization: 91, memory: 83, temperature: 76 },
                      ]}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="utilization" stroke="#8884d8" name="GPU Utilization %" />
                        <Line type="monotone" dataKey="memory" stroke="#82ca9d" name="Memory Usage %" />
                        <Line type="monotone" dataKey="temperature" stroke="#ff7300" name="Temperature °C" />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                {/* Processing Pipeline */}
                <Card className="bg-gray-800 border-gray-700">
                  <CardHeader>
                    <CardTitle className="text-white">Processing Pipeline Status</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {[
                        { name: 'Data Ingestion', status: 'healthy', latency: '12ms' },
                        { name: 'GPU Processing', status: 'healthy', latency: '8ms' },
                        { name: 'ML Inference', status: 'healthy', latency: '15ms' },
                        { name: 'Cache Update', status: 'healthy', latency: '3ms' },
                        { name: 'WebSocket Push', status: 'healthy', latency: '5ms' },
                      ].map((stage, index) => (
                        <div key={stage.name} className="flex items-center justify-between p-3 bg-gray-700 rounded">
                          <div className="flex items-center">
                            <div className={`w-3 h-3 rounded-full mr-3 ${
                              stage.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                            }`} />
                            <span>{stage.name}</span>
                          </div>
                          <Badge variant="outline" className="text-white border-gray-500">
                            {stage.latency}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Cluster Resources */}
                <Card className="bg-gray-800 border-gray-700 lg:col-span-2">
                  <CardHeader>
                    <CardTitle className="text-white">Kubernetes Cluster Status</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                      {[
                        { name: 'GPU Nodes', value: '3/3', status: 'healthy' },
                        { name: 'CPU Nodes', value: '5/5', status: 'healthy' },
                        { name: 'Active Pods', value: '24/30', status: 'healthy' },
                        { name: 'GPU Utilization', value: '87%', status: 'warning' },
                      ].map((metric) => (
                        <div key={metric.name} className="text-center p-4 bg-gray-700 rounded">
                          <div className={`text-2xl font-bold ${
                            metric.status === 'healthy' ? 'text-green-400' : 
                            metric.status === 'warning' ? 'text-yellow-400' : 'text-red-400'
                          }`}>
                            {metric.value}
                          </div>
                          <div className="text-sm text-gray-400">{metric.name}</div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        {/* Alerts */}
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