"""
Adaptive Rate Limiting System
Dynamically adjusts API call delays based on success/failure rates.
"""
import time
import logging
from collections import deque
from typing import Tuple

logger = logging.getLogger(__name__)


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts delays based on API success/failure.
    
    Features:
    - Increases delay when hitting rate limits (429 errors)
    - Decreases delay after consecutive successes
    - Maintains min/max delay bounds
    - Tracks success streaks
    """
    
    def __init__(
        self,
        base_delay: float,
        slowdown_factor: float = 1.5,
        speedup_factor: float = 0.9,
        min_delay: float = 1.0,
        max_delay: float = 10.0,
        success_threshold: int = 5
    ):
        """
        Initialize adaptive rate limiter.
        
        Args:
            base_delay (float): Initial delay in seconds
            slowdown_factor (float): Multiply delay by this on error (default: 1.5)
            speedup_factor (float): Multiply delay by this on success (default: 0.9)
            min_delay (float): Minimum allowed delay (default: 1.0s)
            max_delay (float): Maximum allowed delay (default: 10.0s)
            success_threshold (int): Consecutive successes before speeding up (default: 5)
        """
        self.current_delay = base_delay
        self.base_delay = base_delay
        self.slowdown_factor = slowdown_factor
        self.speedup_factor = speedup_factor
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.success_threshold = success_threshold
        
        # State tracking
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.total_adjustments = 0
        
        # History tracking (last 50 events)
        self.recent_events = deque(maxlen=50)  # True = success, False = failure
        
        logger.info(f"⚡ Adaptive rate limiter initialized:")
        logger.info(f"   Base delay: {base_delay}s")
        logger.info(f"   Range: {min_delay}s - {max_delay}s")
        logger.info(f"   Slowdown factor: {slowdown_factor}x")
        logger.info(f"   Speedup factor: {speedup_factor}x")
        logger.info(f"   Success threshold: {success_threshold}")
    
    def wait(self):
        """Wait for the current delay duration."""
        time.sleep(self.current_delay)
    
    def record_success(self):
        """
        Record a successful API call.
        May decrease delay if enough consecutive successes.
        """
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.recent_events.append(True)
        
        # Speed up after threshold consecutive successes
        if self.consecutive_successes >= self.success_threshold:
            old_delay = self.current_delay
            self.current_delay = max(
                self.min_delay,
                self.current_delay * self.speedup_factor
            )
            
            if old_delay != self.current_delay:
                self.total_adjustments += 1
                logger.info(f"⚡ Rate limit decreased: {old_delay:.2f}s → {self.current_delay:.2f}s (after {self.consecutive_successes} successes)")
                self.consecutive_successes = 0  # Reset counter
    
    def record_failure(self, is_rate_limit_error: bool = False):
        """
        Record a failed API call.
        Increases delay, more aggressively for rate limit errors.
        
        Args:
            is_rate_limit_error (bool): True if 429 error (rate limit)
        """
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.recent_events.append(False)
        
        old_delay = self.current_delay
        
        # More aggressive slowdown for rate limit errors
        if is_rate_limit_error:
            # Double the slowdown factor for 429 errors
            self.current_delay = min(
                self.max_delay,
                self.current_delay * (self.slowdown_factor * 1.5)
            )
            logger.warning(f"🚨 Rate limit hit! Increasing delay: {old_delay:.2f}s → {self.current_delay:.2f}s")
        else:
            # Normal slowdown for other errors
            self.current_delay = min(
                self.max_delay,
                self.current_delay * self.slowdown_factor
            )
            logger.warning(f"⚠️  API error. Increasing delay: {old_delay:.2f}s → {self.current_delay:.2f}s")
        
        if old_delay != self.current_delay:
            self.total_adjustments += 1
    
    def get_current_delay(self) -> float:
        """Get current delay value."""
        return self.current_delay
    
    def get_success_rate(self) -> float:
        """Get recent success rate as percentage."""
        if not self.recent_events:
            return 0.0
        successes = sum(1 for event in self.recent_events if event)
        return (successes / len(self.recent_events)) * 100
    
    def reset(self):
        """Reset to base delay and clear history."""
        old_delay = self.current_delay
        self.current_delay = self.base_delay
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.recent_events.clear()
        logger.info(f"🔄 Rate limiter reset: {old_delay:.2f}s → {self.current_delay:.2f}s")
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            'current_delay': self.current_delay,
            'base_delay': self.base_delay,
            'consecutive_successes': self.consecutive_successes,
            'consecutive_failures': self.consecutive_failures,
            'total_adjustments': self.total_adjustments,
            'recent_success_rate': self.get_success_rate(),
            'at_min_limit': self.current_delay == self.min_delay,
            'at_max_limit': self.current_delay == self.max_delay
        }
    
    def __str__(self):
        """String representation of current state."""
        return (
            f"AdaptiveRateLimiter(delay={self.current_delay:.2f}s, "
            f"successes={self.consecutive_successes}, "
            f"failures={self.consecutive_failures}, "
            f"rate={self.get_success_rate():.1f}%)"
        )


# Global limiter instance (singleton)
_global_limiter = None


def get_rate_limiter() -> AdaptiveRateLimiter:
    """Get the global adaptive rate limiter instance."""
    global _global_limiter
    if _global_limiter is None:
        from config.settings import (
            GEMINI_BUFFER_DELAY,
            ADAPTIVE_SLOWDOWN_FACTOR,
            ADAPTIVE_SPEEDUP_FACTOR,
            ADAPTIVE_MIN_DELAY,
            ADAPTIVE_MAX_DELAY,
            ADAPTIVE_SUCCESS_THRESHOLD
        )
        _global_limiter = AdaptiveRateLimiter(
            base_delay=GEMINI_BUFFER_DELAY,
            slowdown_factor=ADAPTIVE_SLOWDOWN_FACTOR,
            speedup_factor=ADAPTIVE_SPEEDUP_FACTOR,
            min_delay=ADAPTIVE_MIN_DELAY,
            max_delay=ADAPTIVE_MAX_DELAY,
            success_threshold=ADAPTIVE_SUCCESS_THRESHOLD
        )
    return _global_limiter


def reset_rate_limiter():
    """Reset the global rate limiter."""
    global _global_limiter
    if _global_limiter is not None:
        _global_limiter.reset()

