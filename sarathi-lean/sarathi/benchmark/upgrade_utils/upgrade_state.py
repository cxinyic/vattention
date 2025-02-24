"""
Module for handling state during model upgrades.
"""

import threading
import logging

logger = logging.getLogger(__name__)

class UpgradeState:
    """
    Shared state for coordinating overlap serving during upgrade.
    
    This class provides thread-safe access to flag variables that track
    the progress of model upgrades in a multi-threaded environment.
    """
    
    def __init__(self):
        """Initialize upgrade state flags."""
        self.preemption_complete = False
        self.weights_loaded = False
        self._lock = threading.Lock()
        
    def set_preemption_complete(self):
        """
        Mark preemption phase as complete.
        
        This method is called after the LLM engine has successfully
        preempted the required number of blocks for the upgrade.
        """
        with self._lock:
            self.preemption_complete = True
            
    def set_weights_loaded(self):
        """
        Mark weight loading phase as complete.
        
        This method is called after the new model weights have been
        successfully loaded into memory.
        """
        with self._lock:
            self.weights_loaded = True
            
    def is_preemption_complete(self):
        """
        Check if preemption phase is complete.
        
        Returns:
            bool: True if preemption is complete, False otherwise
        """
        with self._lock:
            return self.preemption_complete
            
    def is_weights_loaded(self):
        """
        Check if weight loading phase is complete.
        
        Returns:
            bool: True if weight loading is complete, False otherwise
        """
        with self._lock:
            return self.weights_loaded