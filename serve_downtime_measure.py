#!/usr/bin/env python3
"""
Serve Downtime Measurement Tool
===============================
Measures downtime during:
  1. Scale-up of an existing serve
  2. Stop and restart of a serve

Usage:
  # Probe mode - continuously probe endpoint and log results
  python serve_downtime_measure.py probe --url http://localhost:8000/health --interval 50

  # Analyze mode - compute downtime from a log file
  python serve_downtime_measure.py analyze --log probe_results.log

Workflow:
  1. Start probing in one terminal
  2. In another terminal, trigger scale-up or restart
  3. Stop probing (Ctrl+C) after serve is back up
  4. Analyze the log to get downtime metrics
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import urllib.request
import urllib.error
import socket


@dataclass
class ProbeResult:
    timestamp_ms: int  # Unix timestamp in milliseconds
    timestamp_iso: str  # Human-readable ISO format
    status: str  # "success" or "failure"
    response_time_ms: float  # Response time in milliseconds
    http_code: Optional[int]  # HTTP status code (None if connection failed)
    error: Optional[str]  # Error message if failed


class ServeProber:
    """Continuously probes a serve endpoint and logs results."""

    def __init__(
        self,
        url: str,
        interval_ms: int = 100,
        timeout_ms: int = 5000,
        log_file: str = "probe_results.log",
    ):
        self.url = url
        self.interval_s = interval_ms / 1000.0
        self.timeout_s = timeout_ms / 1000.0
        self.log_file = log_file
        self.running = True
        self.results: list[ProbeResult] = []

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n[INFO] Stopping probe...")
        self.running = False

    def probe_once(self) -> ProbeResult:
        """Send a single probe request and return the result."""
        timestamp_ms = int(time.time() * 1000)
        timestamp_iso = datetime.now().isoformat(timespec="milliseconds")
        start = time.perf_counter()

        try:
            req = urllib.request.Request(self.url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                response_time_ms = (time.perf_counter() - start) * 1000
                http_code = response.getcode()

                if 200 <= http_code < 300:
                    return ProbeResult(
                        timestamp_ms=timestamp_ms,
                        timestamp_iso=timestamp_iso,
                        status="success",
                        response_time_ms=response_time_ms,
                        http_code=http_code,
                        error=None,
                    )
                else:
                    return ProbeResult(
                        timestamp_ms=timestamp_ms,
                        timestamp_iso=timestamp_iso,
                        status="failure",
                        response_time_ms=response_time_ms,
                        http_code=http_code,
                        error=f"Non-2xx response: {http_code}",
                    )

        except urllib.error.HTTPError as e:
            response_time_ms = (time.perf_counter() - start) * 1000
            return ProbeResult(
                timestamp_ms=timestamp_ms,
                timestamp_iso=timestamp_iso,
                status="failure",
                response_time_ms=response_time_ms,
                http_code=e.code,
                error=f"HTTP {e.code}: {e.reason}",
            )

        except urllib.error.URLError as e:
            response_time_ms = (time.perf_counter() - start) * 1000
            return ProbeResult(
                timestamp_ms=timestamp_ms,
                timestamp_iso=timestamp_iso,
                status="failure",
                response_time_ms=response_time_ms,
                http_code=None,
                error=f"Connection error: {e.reason}",
            )

        except socket.timeout:
            response_time_ms = (time.perf_counter() - start) * 1000
            return ProbeResult(
                timestamp_ms=timestamp_ms,
                timestamp_iso=timestamp_iso,
                status="failure",
                response_time_ms=response_time_ms,
                http_code=None,
                error="Request timeout",
            )

        except Exception as e:
            response_time_ms = (time.perf_counter() - start) * 1000
            return ProbeResult(
                timestamp_ms=timestamp_ms,
                timestamp_iso=timestamp_iso,
                status="failure",
                response_time_ms=response_time_ms,
                http_code=None,
                error=str(e),
            )

    def run(self):
        """Run continuous probing until interrupted."""
        print(f"[INFO] Starting probe to {self.url}")
        print(f"[INFO] Interval: {self.interval_s * 1000:.0f}ms, Timeout: {self.timeout_s * 1000:.0f}ms")
        print(f"[INFO] Logging to: {self.log_file}")
        print("[INFO] Press Ctrl+C to stop\n")

        # Write CSV header
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp_ms", "timestamp_iso", "status", "response_time_ms", "http_code", "error"]
            )

        probe_count = 0
        success_count = 0
        failure_count = 0

        while self.running:
            result = self.probe_once()
            self.results.append(result)
            probe_count += 1

            if result.status == "success":
                success_count += 1
                status_char = "✓"
            else:
                failure_count += 1
                status_char = "✗"

            # Log to file
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        result.timestamp_ms,
                        result.timestamp_iso,
                        result.status,
                        f"{result.response_time_ms:.2f}",
                        result.http_code or "",
                        result.error or "",
                    ]
                )

            # Print to console
            error_str = f" ({result.error})" if result.error else ""
            print(
                f"[{result.timestamp_iso}] {status_char} {result.status.upper():7} "
                f"| {result.response_time_ms:7.2f}ms | HTTP {result.http_code or 'N/A':>3}{error_str}"
            )

            # Sleep for the remaining interval time
            elapsed = result.response_time_ms / 1000.0
            sleep_time = max(0, self.interval_s - elapsed)
            if sleep_time > 0 and self.running:
                time.sleep(sleep_time)

        print(f"\n[INFO] Probe stopped. Total: {probe_count}, Success: {success_count}, Failure: {failure_count}")
        print(f"[INFO] Results saved to: {self.log_file}")


@dataclass
class DowntimeWindow:
    """Represents a window of downtime."""
    start_ms: int  # First failure timestamp
    end_ms: int  # First success after failures
    duration_ms: int  # end_ms - start_ms
    failure_count: int  # Number of failed probes in this window


def analyze_log(log_file: str) -> dict:
    """Analyze a probe log file and compute downtime metrics."""
    results = []

    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(
                ProbeResult(
                    timestamp_ms=int(row["timestamp_ms"]),
                    timestamp_iso=row["timestamp_iso"],
                    status=row["status"],
                    response_time_ms=float(row["response_time_ms"]),
                    http_code=int(row["http_code"]) if row["http_code"] else None,
                    error=row["error"] if row["error"] else None,
                )
            )

    if not results:
        return {"error": "No results found in log file"}

    # Find all downtime windows
    downtime_windows: list[DowntimeWindow] = []
    in_downtime = False
    downtime_start_ms = 0
    failure_count = 0

    for i, result in enumerate(results):
        if result.status == "failure":
            if not in_downtime:
                # Start of a new downtime window
                in_downtime = True
                downtime_start_ms = result.timestamp_ms
                failure_count = 1
            else:
                failure_count += 1
        else:  # success
            if in_downtime:
                # End of downtime window
                downtime_end_ms = result.timestamp_ms
                duration_ms = downtime_end_ms - downtime_start_ms
                downtime_windows.append(
                    DowntimeWindow(
                        start_ms=downtime_start_ms,
                        end_ms=downtime_end_ms,
                        duration_ms=duration_ms,
                        failure_count=failure_count,
                    )
                )
                in_downtime = False
                failure_count = 0

    # Handle case where log ends during downtime
    if in_downtime:
        last_result = results[-1]
        downtime_windows.append(
            DowntimeWindow(
                start_ms=downtime_start_ms,
                end_ms=last_result.timestamp_ms,
                duration_ms=last_result.timestamp_ms - downtime_start_ms,
                failure_count=failure_count,
            )
        )

    # Compute statistics
    total_probes = len(results)
    success_count = sum(1 for r in results if r.status == "success")
    failure_count = total_probes - success_count
    total_duration_ms = results[-1].timestamp_ms - results[0].timestamp_ms if len(results) > 1 else 0

    # Response time stats for successful requests
    success_times = [r.response_time_ms for r in results if r.status == "success"]
    avg_response_time = sum(success_times) / len(success_times) if success_times else 0
    max_response_time = max(success_times) if success_times else 0
    min_response_time = min(success_times) if success_times else 0

    return {
        "summary": {
            "log_file": log_file,
            "total_probes": total_probes,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate_percent": (success_count / total_probes * 100) if total_probes > 0 else 0,
            "total_measurement_duration_ms": total_duration_ms,
            "first_probe_time": results[0].timestamp_iso if results else None,
            "last_probe_time": results[-1].timestamp_iso if results else None,
        },
        "response_times_ms": {
            "avg": round(avg_response_time, 2),
            "min": round(min_response_time, 2),
            "max": round(max_response_time, 2),
        },
        "downtime_windows": [
            {
                "window_number": i + 1,
                "start_ms": w.start_ms,
                "end_ms": w.end_ms,
                "duration_ms": w.duration_ms,
                "duration_seconds": round(w.duration_ms / 1000, 3),
                "failure_count": w.failure_count,
            }
            for i, w in enumerate(downtime_windows)
        ],
        "total_downtime_ms": sum(w.duration_ms for w in downtime_windows),
        "total_downtime_seconds": round(sum(w.duration_ms for w in downtime_windows) / 1000, 3),
        "downtime_window_count": len(downtime_windows),
    }


def print_analysis(analysis: dict):
    """Pretty print the analysis results."""
    if "error" in analysis:
        print(f"[ERROR] {analysis['error']}")
        return

    print("\n" + "=" * 70)
    print("SERVE DOWNTIME ANALYSIS REPORT")
    print("=" * 70)

    summary = analysis["summary"]
    print(f"\n📊 SUMMARY")
    print(f"   Log file:           {summary['log_file']}")
    print(f"   Total probes:       {summary['total_probes']}")
    print(f"   Successful:         {summary['success_count']}")
    print(f"   Failed:             {summary['failure_count']}")
    print(f"   Success rate:       {summary['success_rate_percent']:.2f}%")
    print(f"   Measurement window: {summary['total_measurement_duration_ms']}ms ({summary['total_measurement_duration_ms']/1000:.2f}s)")
    print(f"   First probe:        {summary['first_probe_time']}")
    print(f"   Last probe:         {summary['last_probe_time']}")

    resp = analysis["response_times_ms"]
    print(f"\n⏱️  RESPONSE TIMES (successful requests)")
    print(f"   Average: {resp['avg']:.2f}ms")
    print(f"   Min:     {resp['min']:.2f}ms")
    print(f"   Max:     {resp['max']:.2f}ms")

    print(f"\n🔴 DOWNTIME ANALYSIS")
    print(f"   Total downtime windows: {analysis['downtime_window_count']}")
    print(f"   Total downtime:         {analysis['total_downtime_ms']}ms ({analysis['total_downtime_seconds']}s)")

    if analysis["downtime_windows"]:
        print(f"\n   Downtime Windows:")
        for w in analysis["downtime_windows"]:
            print(f"   ├── Window #{w['window_number']}")
            print(f"   │   Duration:      {w['duration_ms']}ms ({w['duration_seconds']}s)")
            print(f"   │   Failed probes: {w['failure_count']}")

    print("\n" + "=" * 70)

    # Output JSON for programmatic use
    json_file = Path(summary["log_file"]).stem + "_analysis.json"
    with open(json_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"[INFO] JSON report saved to: {json_file}")


def compare_scenarios(log_file_1: str, log_file_2: str, label_1: str = "Scale-up", label_2: str = "Restart"):
    """Compare downtime between two scenarios."""
    analysis_1 = analyze_log(log_file_1)
    analysis_2 = analyze_log(log_file_2)

    print("\n" + "=" * 70)
    print("SCENARIO COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<30} | {label_1:>15} | {label_2:>15}")
    print("-" * 70)

    dt1 = analysis_1["total_downtime_ms"]
    dt2 = analysis_2["total_downtime_ms"]

    print(f"{'Total Downtime (ms)':<30} | {dt1:>15} | {dt2:>15}")
    print(f"{'Total Downtime (s)':<30} | {dt1/1000:>15.3f} | {dt2/1000:>15.3f}")
    print(f"{'Downtime Windows':<30} | {analysis_1['downtime_window_count']:>15} | {analysis_2['downtime_window_count']:>15}")
    print(f"{'Total Probes':<30} | {analysis_1['summary']['total_probes']:>15} | {analysis_2['summary']['total_probes']:>15}")
    print(f"{'Failed Probes':<30} | {analysis_1['summary']['failure_count']:>15} | {analysis_2['summary']['failure_count']:>15}")
    print(f"{'Success Rate (%)':<30} | {analysis_1['summary']['success_rate_percent']:>15.2f} | {analysis_2['summary']['success_rate_percent']:>15.2f}")

    print("\n" + "-" * 70)
    if dt1 < dt2:
        diff = dt2 - dt1
        print(f"✅ {label_1} is faster by {diff}ms ({diff/1000:.3f}s)")
        print(f"   {label_1} has {(1 - dt1/dt2)*100:.1f}% less downtime than {label_2}")
    elif dt2 < dt1:
        diff = dt1 - dt2
        print(f"✅ {label_2} is faster by {diff}ms ({diff/1000:.3f}s)")
        print(f"   {label_2} has {(1 - dt2/dt1)*100:.1f}% less downtime than {label_1}")
    else:
        print(f"⚖️  Both scenarios have equal downtime")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Measure serve downtime during scale-up or restart operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start probing an endpoint
  %(prog)s probe --url http://localhost:8000/health --interval 50 --log scale_up.log

  # Analyze results from a single log
  %(prog)s analyze --log scale_up.log

  # Compare two scenarios
  %(prog)s compare --log1 scale_up.log --log2 restart.log

Workflow for measuring downtime:
  1. Terminal 1: python %(prog)s probe --url <serve_url> --log scenario1.log
  2. Terminal 2: Trigger scale-up or restart
  3. Terminal 1: Ctrl+C after serve recovers
  4. Repeat for second scenario with different log file
  5. Compare: python %(prog)s compare --log1 scale_up.log --log2 restart.log
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Probe subcommand
    probe_parser = subparsers.add_parser("probe", help="Continuously probe an endpoint")
    probe_parser.add_argument("--url", required=True, help="URL to probe (e.g., http://localhost:8000/health)")
    probe_parser.add_argument("--interval", type=int, default=100, help="Probe interval in milliseconds (default: 100)")
    probe_parser.add_argument("--timeout", type=int, default=5000, help="Request timeout in milliseconds (default: 5000)")
    probe_parser.add_argument("--log", default="probe_results.log", help="Output log file (default: probe_results.log)")

    # Analyze subcommand
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a probe log file")
    analyze_parser.add_argument("--log", required=True, help="Log file to analyze")
    analyze_parser.add_argument("--json", action="store_true", help="Output only JSON")

    # Compare subcommand
    compare_parser = subparsers.add_parser("compare", help="Compare two scenario logs")
    compare_parser.add_argument("--log1", required=True, help="First log file (e.g., scale_up.log)")
    compare_parser.add_argument("--log2", required=True, help="Second log file (e.g., restart.log)")
    compare_parser.add_argument("--label1", default="Scale-up", help="Label for first scenario")
    compare_parser.add_argument("--label2", default="Restart", help="Label for second scenario")

    args = parser.parse_args()

    if args.command == "probe":
        prober = ServeProber(
            url=args.url,
            interval_ms=args.interval,
            timeout_ms=args.timeout,
            log_file=args.log,
        )
        prober.run()

    elif args.command == "analyze":
        analysis = analyze_log(args.log)
        if args.json:
            print(json.dumps(analysis, indent=2))
        else:
            print_analysis(analysis)

    elif args.command == "compare":
        compare_scenarios(args.log1, args.log2, args.label1, args.label2)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
