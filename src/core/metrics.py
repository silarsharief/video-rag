"""
Metrics and Monitoring System
Tracks API performance, success/failure rates, and system statistics.
"""
import time
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks system metrics and performance statistics.
    
    Metrics tracked:
    - API success/failure rates
    - Retry attempts per batch
    - Processing time per video
    - Query rewrite cache hits/misses
    - Adaptive rate limiting adjustments
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        # API call metrics
        self.total_api_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retry_attempts = defaultdict(int)  # {attempt_number: count}
        
        # Processing metrics
        self.videos_processed = 0
        self.total_processing_time = 0.0
        self.batches_processed = 0
        self.batches_succeeded_first_try = 0
        self.batches_needed_retry = 0
        self.batches_failed = 0
        
        # Query cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Adaptive rate limiting metrics
        self.rate_limit_adjustments = 0
        self.current_delay = 0.0
        self.delay_history = deque(maxlen=100)  # Track last 100 delays
        
        # Error tracking
        self.errors_by_type = defaultdict(int)  # {error_type: count}
        self.recent_errors = deque(maxlen=10)  # Track last 10 errors
        
        # Timestamps
        self.start_time = time.time()
        self.last_log_time = time.time()
    
    def record_api_call(self, success: bool, attempts: int = 1, error_type: Optional[str] = None):
        """
        Record an API call result.
        
        Args:
            success (bool): Whether the call succeeded
            attempts (int): Number of attempts needed
            error_type (str, optional): Type of error if failed
        """
        self.total_api_calls += 1
        
        if success:
            self.successful_calls += 1
            if attempts == 1:
                self.batches_succeeded_first_try += 1
            else:
                self.batches_needed_retry += 1
        else:
            self.failed_calls += 1
            self.batches_failed += 1
            if error_type:
                self.errors_by_type[error_type] += 1
                self.recent_errors.append({
                    'type': error_type,
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'attempts': attempts
                })
        
        self.retry_attempts[attempts] += 1
        self.batches_processed += 1
    
    def record_cache_access(self, hit: bool):
        """
        Record a cache access (hit or miss).
        
        Args:
            hit (bool): True if cache hit, False if cache miss
        """
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def record_rate_limit_adjustment(self, new_delay: float):
        """
        Record a rate limit adjustment.
        
        Args:
            new_delay (float): New delay value in seconds
        """
        self.rate_limit_adjustments += 1
        self.current_delay = new_delay
        self.delay_history.append(new_delay)
    
    def record_video_processing(self, processing_time: float):
        """
        Record video processing completion.
        
        Args:
            processing_time (float): Time taken to process video in seconds
        """
        self.videos_processed += 1
        self.total_processing_time += processing_time
    
    def get_success_rate(self) -> float:
        """Get API success rate as percentage."""
        if self.total_api_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_api_calls) * 100
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total_accesses = self.cache_hits + self.cache_misses
        if total_accesses == 0:
            return 0.0
        return (self.cache_hits / total_accesses) * 100
    
    def get_average_retry_attempts(self) -> float:
        """Get average number of retry attempts per batch."""
        if self.batches_processed == 0:
            return 0.0
        total_attempts = sum(attempts * count for attempts, count in self.retry_attempts.items())
        return total_attempts / self.batches_processed
    
    def get_average_processing_time(self) -> float:
        """Get average processing time per video in seconds."""
        if self.videos_processed == 0:
            return 0.0
        return self.total_processing_time / self.videos_processed
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self.start_time
    
    def should_log_metrics(self, interval: int = 10) -> bool:
        """
        Check if metrics should be logged based on interval.
        
        Args:
            interval (int): Log interval in seconds
            
        Returns:
            bool: True if should log now
        """
        current_time = time.time()
        if current_time - self.last_log_time >= interval:
            self.last_log_time = current_time
            return True
        return False
    
    def log_current_metrics(self):
        """Log current metrics summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 METRICS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"⏱️  Uptime: {self.get_uptime():.1f}s")
        logger.info(f"🎬 Videos Processed: {self.videos_processed}")
        logger.info(f"📞 API Calls: {self.total_api_calls} (✅ {self.successful_calls}, ❌ {self.failed_calls})")
        logger.info(f"✅ Success Rate: {self.get_success_rate():.1f}%")
        logger.info(f"🔁 Avg Retry Attempts: {self.get_average_retry_attempts():.2f}")
        logger.info(f"📦 Batches: {self.batches_succeeded_first_try} first-try, {self.batches_needed_retry} retried, {self.batches_failed} failed")
        
        if self.cache_hits + self.cache_misses > 0:
            logger.info(f"💾 Cache Hit Rate: {self.get_cache_hit_rate():.1f}% ({self.cache_hits} hits, {self.cache_misses} misses)")
        
        if self.rate_limit_adjustments > 0:
            logger.info(f"⚡ Rate Limit Adjustments: {self.rate_limit_adjustments} (current delay: {self.current_delay:.2f}s)")
        
        if self.errors_by_type:
            logger.info(f"⚠️  Errors by Type:")
            for error_type, count in sorted(self.errors_by_type.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"   - {error_type}: {count}")
        
        logger.info(f"{'='*60}\n")
    
    def get_summary_report(self) -> str:
        """
        Generate a comprehensive summary report.
        
        Returns:
            str: Formatted summary report
        """
        report = []
        report.append(f"\n{'='*60}")
        report.append(f"📊 FINAL METRICS REPORT")
        report.append(f"{'='*60}")
        report.append(f"")
        report.append(f"⏱️  SESSION DURATION")
        report.append(f"   Total Uptime: {self.get_uptime():.1f}s ({self.get_uptime()/60:.1f} minutes)")
        report.append(f"")
        report.append(f"🎬 VIDEO PROCESSING")
        report.append(f"   Videos Processed: {self.videos_processed}")
        if self.videos_processed > 0:
            report.append(f"   Avg Processing Time: {self.get_average_processing_time():.1f}s per video")
        report.append(f"")
        report.append(f"📞 API PERFORMANCE")
        report.append(f"   Total API Calls: {self.total_api_calls}")
        report.append(f"   Successful: {self.successful_calls} ({self.get_success_rate():.1f}%)")
        report.append(f"   Failed: {self.failed_calls}")
        report.append(f"   Average Retry Attempts: {self.get_average_retry_attempts():.2f}")
        report.append(f"")
        report.append(f"📦 BATCH PROCESSING")
        report.append(f"   Total Batches: {self.batches_processed}")
        report.append(f"   ✅ First-Try Success: {self.batches_succeeded_first_try} ({self.batches_succeeded_first_try/max(self.batches_processed,1)*100:.1f}%)")
        report.append(f"   🔁 Required Retry: {self.batches_needed_retry} ({self.batches_needed_retry/max(self.batches_processed,1)*100:.1f}%)")
        report.append(f"   ❌ Failed: {self.batches_failed} ({self.batches_failed/max(self.batches_processed,1)*100:.1f}%)")
        
        if self.cache_hits + self.cache_misses > 0:
            report.append(f"")
            report.append(f"💾 CACHE PERFORMANCE")
            report.append(f"   Total Accesses: {self.cache_hits + self.cache_misses}")
            report.append(f"   Cache Hits: {self.cache_hits} ({self.get_cache_hit_rate():.1f}%)")
            report.append(f"   Cache Misses: {self.cache_misses}")
        
        if self.rate_limit_adjustments > 0:
            report.append(f"")
            report.append(f"⚡ ADAPTIVE RATE LIMITING")
            report.append(f"   Adjustments Made: {self.rate_limit_adjustments}")
            report.append(f"   Final Delay: {self.current_delay:.2f}s")
            if self.delay_history:
                report.append(f"   Avg Delay: {sum(self.delay_history)/len(self.delay_history):.2f}s")
                report.append(f"   Min Delay: {min(self.delay_history):.2f}s")
                report.append(f"   Max Delay: {max(self.delay_history):.2f}s")
        
        if self.errors_by_type:
            report.append(f"")
            report.append(f"⚠️  ERROR SUMMARY")
            for error_type, count in sorted(self.errors_by_type.items(), key=lambda x: x[1], reverse=True):
                report.append(f"   - {error_type}: {count} occurrence(s)")
        
        if self.recent_errors:
            report.append(f"")
            report.append(f"🔴 RECENT ERRORS (Last {len(self.recent_errors)})")
            for error in self.recent_errors:
                report.append(f"   [{error['time']}] {error['type']} (attempt {error['attempts']})")
        
        # Performance assessment
        report.append(f"")
        report.append(f"🎯 PERFORMANCE ASSESSMENT")
        success_rate = self.get_success_rate()
        if success_rate >= 95:
            report.append(f"   Status: ✅ EXCELLENT - System performing optimally")
        elif success_rate >= 85:
            report.append(f"   Status: ✅ GOOD - Minor issues detected")
        elif success_rate >= 70:
            report.append(f"   Status: ⚠️  FAIR - Consider tuning rate limits")
        else:
            report.append(f"   Status: ❌ POOR - Requires immediate attention")
        
        report.append(f"{'='*60}\n")
        
        return "\n".join(report)


# Global metrics instance (singleton)
_global_metrics = None


def get_metrics() -> MetricsTracker:
    """Get the global metrics tracker instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsTracker()
    return _global_metrics


def reset_metrics():
    """Reset the global metrics tracker."""
    global _global_metrics
    _global_metrics = MetricsTracker()

