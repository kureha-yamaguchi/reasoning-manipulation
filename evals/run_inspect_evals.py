#!/usr/bin/env python3
"""Run Inspect AI evals against a vLLM-served model.

Starts a vLLM server for the given model, runs the specified Inspect
benchmarks, and stores logs under results/{model_label}/eval_results/.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_TASKS = {
    "aime2025": "inspect_evals/aime2025",
    "gpqa_diamond": "inspect_evals/gpqa_diamond",
    "math": "inspect_evals/mathematics",
}

# Extra CLI flags for benchmarks that need them
BENCHMARK_EXTRA_ARGS = {
    "math": ["--limit", "500"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Inspect evals via vLLM")
    parser.add_argument(
        "--model_path",
        required=True,
        help="HuggingFace model ID or local path to model weights",
    )
    parser.add_argument(
        "--model_label",
        required=True,
        help="Label used for results directory naming (e.g. 'Qwen3-8B_base')",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARK_TASKS.keys()),
        choices=list(BENCHMARK_TASKS.keys()),
        help="Which benchmarks to run",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for the vLLM OpenAI-compatible server",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=40960,
        help="Maximum model context length for vLLM (Qwen3-8B max is 40960)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help="Max tokens for generation",
    )
    return parser.parse_args()


def kill_process_on_port(port: int):
    """Kill any existing process listening on the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                print(f"Killing existing process {pid} on port {port}", flush=True)
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
    except FileNotFoundError:
        # lsof not available, try fuser
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)

    # Wait for port to actually be released
    for _ in range(30):
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return  # port is free
        except OSError:
            time.sleep(1)
    print(f"WARNING: port {port} may still be in use", flush=True)


def build_vllm_cmd(args) -> list[str]:
    """Build the vLLM server launch command."""
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model_path,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--served-model-name",
        args.model_label,
    ]
    return cmd


def wait_for_server(port: int, timeout: int = 600, interval: int = 10) -> bool:
    """Wait for the vLLM server to be healthy."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        print(f"  Waiting for vLLM server on port {port}... ({int(time.time() - start)}s elapsed)")
        time.sleep(interval)
    return False


def run_inspect_eval(
    benchmark: str,
    task_name: str,
    model_label: str,
    port: int,
    temperature: float,
    max_tokens: int,
) -> bool:
    """Run a single Inspect eval benchmark."""
    log_dir = PROJECT_ROOT / "results" / model_label / "eval_results"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = f"http://localhost:{port}/v1"
    env["OPENAI_API_KEY"] = "dummy"
    env["INSPECT_LOG_DIR"] = str(log_dir)

    cmd = [
        sys.executable,
        "-m",
        "inspect_ai",
        "eval",
        task_name,
        "--model",
        f"openai/{model_label}",
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
    ]

    extra = BENCHMARK_EXTRA_ARGS.get(benchmark, [])
    cmd.extend(extra)

    print(f"\n{'='*60}")
    print(f"Running benchmark: {benchmark}")
    print(f"  Task: {task_name}")
    print(f"  Log dir: {log_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"WARNING: Benchmark {benchmark} exited with code {result.returncode}")
        return False
    return True


def main():

    print("Starting run_inspect_evals.py")

    args = parse_args()

    print(f"Model path:  {args.model_path}")
    print(f"Model label: {args.model_label}")
    print(f"Benchmarks:  {args.benchmarks}")
    print(f"Port:        {args.port}")

    # Kill any leftover vLLM server on the port
    kill_process_on_port(args.port)

    # Start vLLM server
    vllm_cmd = build_vllm_cmd(args)
    print(f"\nStarting vLLM server: {' '.join(vllm_cmd)}\n", flush=True)

    log_dir = PROJECT_ROOT / "results" / args.model_label / "eval_results"
    log_dir.mkdir(parents=True, exist_ok=True)
    vllm_log_path = log_dir / "vllm_server.log"
    vllm_log = open(vllm_log_path, "w")
    print(f"vLLM server logs: {vllm_log_path}", flush=True)

    vllm_proc = subprocess.Popen(
        vllm_cmd,
        stdout=vllm_log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    try:
        # Stream vLLM logs to terminal while waiting for startup
        tail_proc = subprocess.Popen(
            ["tail", "-f", str(vllm_log_path)],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        print("Waiting for vLLM server to become healthy...", flush=True)
        healthy = wait_for_server(args.port)

        # Stop streaming vLLM logs once server is up (or failed)
        tail_proc.terminate()
        tail_proc.wait()

        if not healthy:
            print("ERROR: vLLM server did not become healthy within timeout.")
            sys.exit(1)
        print("vLLM server is ready!\n", flush=True)

        results = {}
        for benchmark in args.benchmarks:
            task_name = BENCHMARK_TASKS[benchmark]
            success = run_inspect_eval(
                benchmark=benchmark,
                task_name=task_name,
                model_label=args.model_label,
                port=args.port,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            results[benchmark] = "PASS" if success else "FAIL"

        print(f"\n{'='*60}")
        print("Eval Summary:")
        for bench, status in results.items():
            print(f"  {bench}: {status}")
        print(f"{'='*60}\n")

    finally:
        # Shut down vLLM server
        print("Shutting down vLLM server...", flush=True)
        try:
            os.killpg(os.getpgid(vllm_proc.pid), signal.SIGTERM)
            vllm_proc.wait(timeout=15)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(vllm_proc.pid), signal.SIGKILL)
                vllm_proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        # Ensure port is fully released before returning
        kill_process_on_port(args.port)
        vllm_log.close()
        print("vLLM server stopped.", flush=True)


if __name__ == "__main__":
    main()
