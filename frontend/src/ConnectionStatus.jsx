// ============================================================================
// frontend/src/components/ConnectionStatus.jsx - Connection Status Indicator
// ============================================================================

import React from 'react';

const ConnectionStatus = ({ status }) => {
  const getStatusConfig = (status) => {
    switch (status) {
      case 'connected':
        return {
          color: 'bg-green-500',
          text: 'Live'
        };
      case 'error':
        return {
          color: 'bg-red-500',
          text: 'Error'
        };
      default:
        return {
          color: 'bg-yellow-500',
          text: 'Connecting...'
        };
    }
  };

  const { color, text } = getStatusConfig(status);

  return (
    <div className="flex items-center space-x-2">
      <div className={`w-3 h-3 rounded-full ${color}`} />
      <span className="text-sm font-medium">{text}</span>
    </div>
  );
};

export default ConnectionStatus;