#!/usr/bin/env python3
"""
Latency Probe for vLLM Scale-Up Testing
========================================
Sends continuous inference requests to measure latency impact during scale-up.

Usage:
  # Send requests every 1 second with 200 token responses
  python3 latency_probe.py --url http://localhost:8006 --response-tokens 200 --interval 1000 --log latency.log

  # Send requests every 500ms with longer (500 token) responses
  python3 latency_probe.py --url http://localhost:8006 --response-tokens 500 --interval 500 --log latency.log

  # Analyze results
  python3 latency_probe.py --analyze --log latency.log

IMPORTANT: Copy to /tmp to avoid being killed by cleanup scripts:
  cp latency_probe.py /tmp/lprobe.py
  cd /tmp
  python3 lprobe.py --url http://localhost:8006 --response-tokens 200 --interval 1000 --log latency.log

Workflow:
  1. Terminal 2: Start latency probe (sends requests at fixed interval)
  2. Terminal 3: Trigger scale-up while requests are in flight
  3. Terminal 2: Observe latency spike, Ctrl+C when done
  4. Analyze: python3 lprobe.py --analyze --log latency.log

Note: With --interval, requests are sent at a fixed rate WITHOUT waiting for
      previous requests to complete. This ensures requests are always in-flight
      when scale-up happens.
"""

import argparse
import csv
import json
import signal
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import urllib.request
import urllib.error


@dataclass
class RequestResult:
    request_id: int
    start_time_iso: str
    end_time_iso: str
    start_time_ms: int
    end_time_ms: int
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    status: str  # "success" or "failure"
    error: Optional[str]


class LatencyProbe:
    """Sends continuous inference requests and measures latency."""

    def __init__(
        self,
        base_url: str,
        response_tokens: int = 100,
        prompt: str = None,
        log_file: str = "latency.log",
        timeout_s: float = 300,  # 5 min timeout for long requests
        interval_ms: int = 1000,  # Send a request every N milliseconds
    ):
        self.base_url = base_url.rstrip("/")
        self.completions_url = f"{self.base_url}/v1/completions"
        self.response_tokens = response_tokens  # Exact number of tokens to generate
        self.prompt = prompt or self._default_prompt()
        self.log_file = log_file
        self.timeout_s = timeout_s
        self.interval_s = interval_ms / 1000.0
        self.running = True
        self.results: list[RequestResult] = []
        self.request_counter = 0
        self.counter_lock = threading.Lock()
        self.file_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.in_flight = 0
        self.in_flight_lock = threading.Lock()
        self.model_name: str | None = None

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _default_prompt(self) -> str:
        """A prompt that encourages longer generation."""
        return (
            "Write a detailed technical explanation about how distributed "
            "computing systems handle fault tolerance and load balancing. "
            "Include specific examples and best practices. "
            "Be thorough and comprehensive in your explanation."
        )

    def _signal_handler(self, signum, frame):
        print("\n[INFO] Stopping latency probe...")
        self.running = False

    def _get_model_name(self) -> str:
        """Get the model name from the server (cached)."""
        if self.model_name:
            return self.model_name
        try:
            url = f"{self.base_url}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                if data.get("data"):
                    self.model_name = data["data"][0]["id"]
                    return self.model_name
        except Exception:
            pass
        return "default"

    def _get_next_request_id(self) -> int:
        """Thread-safe request ID generation."""
        with self.counter_lock:
            self.request_counter += 1
            return self.request_counter

    def _send_request_async(self, request_id: int):
        """Send a request in a background thread and log when complete."""
        with self.in_flight_lock:
            self.in_flight += 1
        
        start_time_ms = int(time.time() * 1000)
        start_time_iso = datetime.now().isoformat(timespec="milliseconds")
        start_perf = time.perf_counter()

        try:
            model_name = self._get_model_name()
            
            payload = {
                "model": model_name,
                "prompt": self.prompt,
                "min_tokens": self.response_tokens,  # Force exact token count
                "max_tokens": self.response_tokens,  # Force exact token count
                "temperature": 0.7,
                "stream": False,
            }
            
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.completions_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                end_perf = time.perf_counter()
                end_time_ms = int(time.time() * 1000)
                end_time_iso = datetime.now().isoformat(timespec="milliseconds")
                latency_ms = (end_perf - start_perf) * 1000
                
                result_data = json.loads(response.read().decode())
                tokens_generated = result_data.get("usage", {}).get("completion_tokens", 0)
                tokens_per_second = (tokens_generated / (latency_ms / 1000)) if latency_ms > 0 else 0
                
                result = RequestResult(
                    request_id=request_id,
                    start_time_iso=start_time_iso,
                    end_time_iso=end_time_iso,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                    latency_ms=latency_ms,
                    tokens_generated=tokens_generated,
                    tokens_per_second=tokens_per_second,
                    status="success",
                    error=None,
                )

        except urllib.error.HTTPError as e:
            end_perf = time.perf_counter()
            end_time_ms = int(time.time() * 1000)
            end_time_iso = datetime.now().isoformat(timespec="milliseconds")
            latency_ms = (end_perf - start_perf) * 1000
            
            result = RequestResult(
                request_id=request_id,
                start_time_iso=start_time_iso,
                end_time_iso=end_time_iso,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                latency_ms=latency_ms,
                tokens_generated=0,
                tokens_per_second=0,
                status="failure",
                error=f"HTTP {e.code}: {e.reason}",
            )

        except Exception as e:
            end_perf = time.perf_counter()
            end_time_ms = int(time.time() * 1000)
            end_time_iso = datetime.now().isoformat(timespec="milliseconds")
            latency_ms = (end_perf - start_perf) * 1000
            
            result = RequestResult(
                request_id=request_id,
                start_time_iso=start_time_iso,
                end_time_iso=end_time_iso,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                latency_ms=latency_ms,
                tokens_generated=0,
                tokens_per_second=0,
                status="failure",
                error=str(e),
            )
        
        # Log result
        self._log_result(result)
        
        with self.in_flight_lock:
            self.in_flight -= 1

    def _log_result(self, result: RequestResult):
        """Thread-safe logging of results."""
        self.results.append(result)
        
        # Log to file
        with self.file_lock:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    result.request_id,
                    result.start_time_iso,
                    result.end_time_iso,
                    result.start_time_ms,
                    result.end_time_ms,
                    f"{result.latency_ms:.2f}",
                    result.tokens_generated,
                    f"{result.tokens_per_second:.2f}",
                    result.status,
                    result.error or "",
                ])

        # Print to console
        with self.print_lock:
            with self.in_flight_lock:
                in_flight = self.in_flight
            if result.status == "success":
                print(
                    f"[{result.end_time_iso}] ✓ #{result.request_id:4d} | "
                    f"Latency: {result.latency_ms:8.2f}ms | "
                    f"Tokens: {result.tokens_generated:4d} | "
                    f"TPS: {result.tokens_per_second:6.2f} | "
                    f"In-flight: {in_flight}"
                )
            else:
                print(
                    f"[{result.end_time_iso}] ✗ #{result.request_id:4d} | "
                    f"Latency: {result.latency_ms:8.2f}ms | "
                    f"FAILED: {result.error} | "
                    f"In-flight: {in_flight}"
                )

    def run(self):
        """Run continuous requests until interrupted."""
        print(f"[INFO] Starting latency probe to {self.base_url}")
        print(f"[INFO] Response tokens (exact): {self.response_tokens}")
        print(f"[INFO] Interval between requests: {self.interval_s * 1000:.0f}ms")
        print(f"[INFO] Logging to: {self.log_file}")
        print("[INFO] Press Ctrl+C to stop")
        print("[INFO] Requests are sent at fixed interval WITHOUT waiting for response\n")

        # Write CSV header
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "request_id", "start_time_iso", "end_time_iso",
                "start_time_ms", "end_time_ms", "latency_ms",
                "tokens_generated", "tokens_per_second", "status", "error"
            ])

        # Send requests at fixed interval
        while self.running:
            request_id = self._get_next_request_id()
            
            # Start request in background thread
            t = threading.Thread(
                target=self._send_request_async,
                args=(request_id,),
                daemon=True
            )
            t.start()
            
            with self.print_lock:
                with self.in_flight_lock:
                    in_flight = self.in_flight
                print(f"[{datetime.now().isoformat(timespec='milliseconds')}] → Sent #{request_id:4d} | In-flight: {in_flight}")
            
            # Wait for interval
            time.sleep(self.interval_s)

        # Wait for in-flight requests to complete
        print("\n[INFO] Waiting for in-flight requests to complete...")
        wait_start = time.time()
        while self.in_flight > 0 and (time.time() - wait_start) < 60:
            time.sleep(0.5)
            with self.in_flight_lock:
                print(f"[INFO] Still waiting... {self.in_flight} requests in-flight")

        # Print summary
        successful = [r for r in self.results if r.status == "success"]
        failed = [r for r in self.results if r.status == "failure"]
        
        print(f"\n[INFO] Probe stopped.")
        print(f"[INFO] Total requests sent: {self.request_counter}")
        print(f"[INFO] Results received: {len(self.results)}")
        print(f"[INFO] Successful: {len(successful)}, Failed: {len(failed)}")
        if successful:
            latencies = [r.latency_ms for r in successful]
            print(f"[INFO] Latency - Avg: {statistics.mean(latencies):.2f}ms, "
                  f"Min: {min(latencies):.2f}ms, Max: {max(latencies):.2f}ms")
        print(f"[INFO] Results saved to: {self.log_file}")


def analyze_log(log_file: str):
    """Analyze latency log and detect scale-up impact."""
    results = []
    
    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "request_id": int(row["request_id"]),
                "start_time_iso": row["start_time_iso"],
                "end_time_iso": row["end_time_iso"],
                "start_time_ms": int(row["start_time_ms"]),
                "end_time_ms": int(row["end_time_ms"]),
                "latency_ms": float(row["latency_ms"]),
                "tokens_generated": int(row["tokens_generated"]),
                "tokens_per_second": float(row["tokens_per_second"]),
                "status": row["status"],
                "error": row["error"] if row["error"] else None,
            })

    if not results:
        print("[ERROR] No results found in log file")
        return

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failure"]

    if not successful:
        print("[ERROR] No successful requests found")
        return

    latencies = [r["latency_ms"] for r in successful]
    
    # Calculate statistics
    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    stdev_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0

    # Detect outliers (latency > 2x median)
    outlier_threshold = median_latency * 2
    outliers = [r for r in successful if r["latency_ms"] > outlier_threshold]

    # Detect the scale-up impact window
    # Look for sudden latency spike
    baseline_latencies = latencies[:max(3, len(latencies)//4)]  # First quarter as baseline
    baseline_avg = statistics.mean(baseline_latencies) if baseline_latencies else avg_latency
    
    spike_requests = [r for r in successful if r["latency_ms"] > baseline_avg * 1.5]

    print("\n" + "=" * 70)
    print("LATENCY ANALYSIS REPORT")
    print("=" * 70)

    print(f"\n📊 SUMMARY")
    print(f"   Log file:           {log_file}")
    print(f"   Total requests:     {len(results)}")
    print(f"   Successful:         {len(successful)}")
    print(f"   Failed:             {len(failed)}")

    print(f"\n⏱️  LATENCY STATISTICS (successful requests)")
    print(f"   Average:            {avg_latency:.2f}ms")
    print(f"   Median:             {median_latency:.2f}ms")
    print(f"   Min:                {min_latency:.2f}ms")
    print(f"   Max:                {max_latency:.2f}ms")
    print(f"   Std Dev:            {stdev_latency:.2f}ms")

    print(f"\n📈 BASELINE vs PEAK")
    print(f"   Baseline (first 25%): {baseline_avg:.2f}ms")
    print(f"   Peak latency:         {max_latency:.2f}ms")
    print(f"   Latency increase:     {max_latency - baseline_avg:.2f}ms ({(max_latency/baseline_avg - 1)*100:.1f}% increase)")

    if outliers:
        print(f"\n⚠️  LATENCY SPIKES (>{outlier_threshold:.0f}ms, 2x median)")
        print(f"   Number of spikes:   {len(outliers)}")
        for r in outliers[:5]:  # Show first 5
            print(f"   - Request #{r['request_id']} at {r['start_time_iso']}: {r['latency_ms']:.2f}ms")
        if len(outliers) > 5:
            print(f"   ... and {len(outliers) - 5} more")

    if spike_requests:
        print(f"\n🎯 SCALE-UP IMPACT (requests with >50% latency increase)")
        print(f"   Affected requests:  {len(spike_requests)}")
        if spike_requests:
            spike_start = spike_requests[0]["start_time_iso"]
            spike_end = spike_requests[-1]["end_time_iso"]
            print(f"   Impact window:      {spike_start} to {spike_end}")
            spike_latencies = [r["latency_ms"] for r in spike_requests]
            print(f"   Avg spike latency:  {statistics.mean(spike_latencies):.2f}ms")

    # Token throughput analysis
    tps_values = [r["tokens_per_second"] for r in successful if r["tokens_per_second"] > 0]
    if tps_values:
        print(f"\n🚀 THROUGHPUT (tokens/second)")
        print(f"   Average TPS:        {statistics.mean(tps_values):.2f}")
        print(f"   Min TPS:            {min(tps_values):.2f}")
        print(f"   Max TPS:            {max(tps_values):.2f}")

    print("\n" + "=" * 70)

    # Save detailed analysis to JSON
    analysis = {
        "summary": {
            "log_file": log_file,
            "total_requests": len(results),
            "successful": len(successful),
            "failed": len(failed),
        },
        "latency_stats": {
            "avg_ms": round(avg_latency, 2),
            "median_ms": round(median_latency, 2),
            "min_ms": round(min_latency, 2),
            "max_ms": round(max_latency, 2),
            "stdev_ms": round(stdev_latency, 2),
        },
        "scale_up_impact": {
            "baseline_latency_ms": round(baseline_avg, 2),
            "peak_latency_ms": round(max_latency, 2),
            "latency_increase_ms": round(max_latency - baseline_avg, 2),
            "latency_increase_percent": round((max_latency/baseline_avg - 1)*100, 1),
            "affected_requests": len(spike_requests),
        },
        "outliers": [
            {"request_id": r["request_id"], "latency_ms": r["latency_ms"]}
            for r in outliers
        ],
    }

    json_file = log_file.replace(".log", "_analysis.json")
    with open(json_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"[INFO] JSON analysis saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure inference latency during scale-up",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send a request every 1 second with 200 token responses
  python3 lprobe.py --url http://localhost:8006 --response-tokens 200 --interval 1000 --log latency.log

  # Send a request every 500ms with 500 token responses
  python3 lprobe.py --url http://localhost:8006 --response-tokens 500 --interval 500 --log latency.log

  # Analyze results
  python3 lprobe.py --analyze --log latency.log
        """,
    )

    parser.add_argument("--url", default="http://localhost:8006",
                        help="Base URL of vLLM server (default: http://localhost:8006)")
    parser.add_argument("--response-tokens", type=int, default=100,
                        help="Exact number of tokens per response (default: 100)")
    parser.add_argument("--interval", type=int, default=1000,
                        help="Interval between requests in ms (default: 1000)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt to use (default: built-in prompt)")
    parser.add_argument("--log", default="latency.log",
                        help="Output log file (default: latency.log)")
    parser.add_argument("--timeout", type=float, default=300,
                        help="Request timeout in seconds (default: 300)")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing log file instead of probing")

    args = parser.parse_args()

    if args.analyze:
        analyze_log(args.log)
    else:
        probe = LatencyProbe(
            base_url=args.url,
            response_tokens=args.response_tokens,
            prompt=args.prompt,
            log_file=args.log,
            timeout_s=args.timeout,
            interval_ms=args.interval,
        )
        probe.run()


if __name__ == "__main__":
    main()
