# ============================================================================
# config/logging_config.py - Detailed Logging Configuration
# ============================================================================

import logging.config
from typing import Dict, Any

def get_logging_config(log_level: str = 'INFO', log_file: str = None) -> Dict[str, Any]:
    """Get detailed logging configuration"""
    
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s'
            },
            'json': {
                'format': '%(asctime)s %(name)s %(levelname)s %(message)s',
                'class': 'pythonjsonlogger.jsonlogger.JsonFormatter'
            }
        },
        'handlers': {
            'console': {
                'level': log_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console'],
                'level': log_level,
                'propagate': False
            },
            'f1_telemetry': {
                'handlers': ['console'],
                'level': log_level,
                'propagate': False
            },
            'uvicorn': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }
    
    # Add file handler if log file specified
    if log_file:
        config['handlers']['file'] = {
            'level': log_level,
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed'
        }
        
        # Add file handler to loggers
        for logger_config in config['loggers'].values():
            logger_config['handlers'].append('file')
    
    return config