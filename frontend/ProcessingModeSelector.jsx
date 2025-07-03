// ============================================================================
// ProcessingModeSelector.jsx - CPU/GPU Mode Selection Component
// ============================================================================

import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Switch } from '@/components/ui/switch';
import { 
  Cpu, 
  Zap, 
  Settings, 
  Info, 
  CheckCircle, 
  AlertTriangle,
  Activity
} from 'lucide-react';

const ProcessingModeSelector = ({ 
  currentMode, 
  onModeChange, 
  systemCapabilities,
  isLoading = false 
}) => {
  const [selectedMode, setSelectedMode] = useState(currentMode);
  const [isApplying, setIsApplying] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Performance characteristics for each mode
  const modeInfo = {
    cpu: {
      icon: Cpu,
      name: 'CPU Mode',
      description: 'Standard processing using CPU only',
      color: 'blue',
      capabilities: {
        maxSamplesPerSec: 100,
        maxDrivers: 20,
        realTimeInference: false,
        complexModels: false,
        batchProcessing: true
      },
      pros: [
        'Universal compatibility',
        'Lower power consumption',
        'Stable performance',
        'No special hardware required'
      ],
      cons: [
        'Slower processing speed',
        'Limited real-time capabilities',
        'Higher latency predictions'
      ]
    },
    gpu: {
      icon: Zap,
      name: 'GPU Mode',
      description: 'High-performance processing with GPU acceleration',
      color: 'green',
      capabilities: {
        maxSamplesPerSec: 50000,
        maxDrivers: 20,
        realTimeInference: true,
        complexModels: true,
        batchProcessing: true
      },
      pros: [
        'Ultra-fast processing',
        'Real-time inference',
        'Large batch processing',
        'Advanced ML models'
      ],
      cons: [
        'Requires NVIDIA GPU',
        'Higher power consumption',
        'More complex setup'
      ]
    }
  };

  // Check system capabilities and recommendations
  const getSystemRecommendation = () => {
    if (!systemCapabilities) return null;

    const { gpu_available, gpu_count, total_memory_gb } = systemCapabilities;

    if (gpu_available && gpu_count > 0) {
      return {
        recommended: 'gpu',
        reason: `GPU acceleration available (${gpu_count} GPU${gpu_count > 1 ? 's' : ''} detected)`,
        confidence: 'high'
      };
    } else {
      return {
        recommended: 'cpu',
        reason: 'No GPU detected - CPU mode recommended',
        confidence: 'medium'
      };
    }
  };

  const recommendation = getSystemRecommendation();

  const handleModeSelect = (mode) => {
    setSelectedMode(mode);
  };

  const handleApplyMode = async () => {
    if (selectedMode === currentMode) return;

    setIsApplying(true);
    try {
      await onModeChange(selectedMode);
    } catch (error) {
      console.error('Failed to change processing mode:', error);
      // Reset to current mode on error
      setSelectedMode(currentMode);
    } finally {
      setIsApplying(false);
    }
  };

  const ModeCard = ({ mode, isSelected, isAvailable }) => {
    const info = modeInfo[mode];
    const Icon = info.icon;
    const isRecommended = recommendation?.recommended === mode;

    return (
      <Card 
        className={`cursor-pointer transition-all duration-200 ${
          isSelected 
            ? `border-${info.color}-500 bg-${info.color}-50 dark:bg-${info.color}-950` 
            : 'border-gray-200 hover:border-gray-300'
        } ${!isAvailable ? 'opacity-50' : ''}`}
        onClick={() => isAvailable && handleModeSelect(mode)}
      >
        <CardContent className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className={`p-2 rounded-lg bg-${info.color}-100 dark:bg-${info.color}-900`}>
                <Icon className={`h-6 w-6 text-${info.color}-600 dark:text-${info.color}-400`} />
              </div>
              <div>
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  {info.name}
                  {isRecommended && (
                    <Badge variant="secondary" className="text-xs">
                      Recommended
                    </Badge>
                  )}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {info.description}
                </p>
              </div>
            </div>
            {isSelected && (
              <CheckCircle className="h-5 w-5 text-green-500" />
            )}
          </div>

          {/* Performance Metrics */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {info.capabilities.maxSamplesPerSec.toLocaleString()}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">
                Samples/sec
              </div>
            </div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {info.capabilities.realTimeInference ? '<50ms' : '1-5s'}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">
                Latency
              </div>
            </div>
          </div>

          {/* Availability Status */}
          {!isAvailable && (
            <Alert className="mb-4">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                {mode === 'gpu' 
                  ? 'GPU not available - install NVIDIA drivers and CUDA'
                  : 'CPU mode not available'
                }
              </AlertDescription>
            </Alert>
          )}

          {/* Advanced Details Toggle */}
          {showAdvanced && (
            <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
              <div className="space-y-3">
                <div>
                  <h4 className="text-sm font-medium text-green-700 dark:text-green-400 mb-1">
                    Advantages
                  </h4>
                  <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                    {info.pros.map((pro, index) => (
                      <li key={index} className="flex items-center">
                        <CheckCircle className="h-3 w-3 text-green-500 mr-2" />
                        {pro}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-orange-700 dark:text-orange-400 mb-1">
                    Limitations
                  </h4>
                  <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                    {info.cons.map((con, index) => (
                      <li key={index} className="flex items-center">
                        <AlertTriangle className="h-3 w-3 text-orange-500 mr-2" />
                        {con}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900">
              <Settings className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <CardTitle>Processing Mode</CardTitle>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Choose between CPU and GPU processing modes
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600">Show Details</span>
              <Switch
                checked={showAdvanced}
                onCheckedChange={setShowAdvanced}
              />
            </div>
            <Badge variant="outline" className="flex items-center gap-1">
              <Activity className="h-3 w-3" />
              Current: {modeInfo[currentMode]?.name}
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* System Recommendation */}
        {recommendation && (
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription>
              <strong>System Recommendation:</strong> {recommendation.reason}
            </AlertDescription>
          </Alert>
        )}

        {/* Mode Selection Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <ModeCard
            mode="cpu"
            isSelected={selectedMode === 'cpu'}
            isAvailable={true} // CPU is always available
          />
          <ModeCard
            mode="gpu"
            isSelected={selectedMode === 'gpu'}
            isAvailable={systemCapabilities?.gpu_available || false}
          />
        </div>

        {/* Performance Comparison Chart */}
        {showAdvanced && (
          <Card className="mt-6">
            <CardHeader>
              <CardTitle className="text-lg">Performance Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  { name: 'Processing Speed', cpu: 20, gpu: 100, unit: '%' },
                  { name: 'Real-time Capability', cpu: 30, gpu: 95, unit: '%' },
                  { name: 'Power Efficiency', cpu: 90, gpu: 60, unit: '%' },
                  { name: 'Setup Complexity', cpu: 10, gpu: 70, unit: '%' }
                ].map((metric) => (
                  <div key={metric.name} className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>{metric.name}</span>
                      <div className="flex space-x-4">
                        <span className="text-blue-600">CPU: {metric.cpu}{metric.unit}</span>
                        <span className="text-green-600">GPU: {metric.gpu}{metric.unit}</span>
                      </div>
                    </div>
                    <div className="flex space-x-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full" 
                          style={{ width: `${metric.cpu}%` }}
                        />
                      </div>
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-green-500 h-2 rounded-full" 
                          style={{ width: `${metric.gpu}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Apply Changes Button */}
        {selectedMode !== currentMode && (
          <div className="flex justify-center pt-4 border-t border-gray-200 dark:border-gray-700">
            <Button
              onClick={handleApplyMode}
              disabled={isApplying || isLoading}
              className="min-w-[200px]"
            >
              {isApplying ? (
                <>
                  <Activity className="h-4 w-4 mr-2 animate-spin" />
                  Switching to {modeInfo[selectedMode]?.name}...
                </>
              ) : (
                <>
                  Apply {modeInfo[selectedMode]?.name}
                </>
              )}
            </Button>
          </div>
        )}

        {/* Mode Change Warning */}
        {selectedMode !== currentMode && (
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription>
              Switching processing modes will restart the telemetry processing pipeline. 
              This may take 10-30 seconds to complete.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default ProcessingModeSelector;