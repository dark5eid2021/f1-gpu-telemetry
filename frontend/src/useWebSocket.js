// ============================================================================
// frontend/src/hooks/useWebSocket.js - WebSocket Management Hook
// ============================================================================

import { useState, useEffect, useRef, useCallback } from 'react';

class WebSocketManager {
  constructor() {
    this.activeConnections = [];
    this.connectionCount = 0;
    this.messageHandler = null;
  }

  setMessageHandler(handler) {
    this.messageHandler = handler;
  }

  async connect(websocket) {
    this.activeConnections.push(websocket);
    this.connectionCount += 1;
    console.log(`📡 WebSocket connected. Total connections: ${this.activeConnections.length}`);
  }

  disconnect(websocket) {
    const index = this.activeConnections.indexOf(websocket);
    if (index > -1) {
      this.activeConnections.splice(index, 1);
    }
    console.log(`🔌 WebSocket disconnected. Total connections: ${this.activeConnections.length}`);
  }

  async broadcast(message) {
    if (!this.activeConnections.length) return;

    const disconnected = [];
    
    for (const connection of this.activeConnections) {
      try {
        await connection.send(message);
      } catch (error) {
        disconnected.push(connection);
      }
    }

    // Remove disconnected clients
    disconnected.forEach(connection => this.disconnect(connection));
  }

  handleMessage(data) {
    if (this.messageHandler) {
      this.messageHandler(data);
    }
  }
}

export const useWebSocket = () => {
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const wsRef = useRef(null);
  const websocketManager = useRef(new WebSocketManager());

  const connectWebSocket = useCallback(() => {
    const wsUrl = process.env.NODE_ENV === 'production' 
      ? 'wss://f1-gpu.yourdomain.com/ws/telemetry'
      : 'ws://localhost:8000/ws/telemetry';
    
    wsRef.current = new WebSocket(wsUrl);
    
    wsRef.current.onopen = () => {
      setConnectionStatus('connected');
      console.log('WebSocket connected');
      websocketManager.current.connect(wsRef.current);
    };
    
    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        websocketManager.current.handleMessage(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    wsRef.current.onclose = () => {
      setConnectionStatus('disconnected');
      console.log('WebSocket disconnected, attempting to reconnect...');
      websocketManager.current.disconnect(wsRef.current);
      setTimeout(connectWebSocket, 3000);
    };
    
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };
  }, []);

  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  return {
    connectionStatus,
    websocketManager: websocketManager.current
  };
};