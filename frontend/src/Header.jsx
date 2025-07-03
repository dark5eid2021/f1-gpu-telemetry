// ============================================================================
// frontend/src/components/Header.jsx - Application Header Component
// ============================================================================

import React from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import ConnectionStatus from './ConnectionStatus';

const Header = ({ connectionStatus }) => {
  return (
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
            <ConnectionStatus status={connectionStatus} />
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
  );
};

export default Header;