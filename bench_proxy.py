"""Benchmark DeepSeek API latency: with-proxy vs without-proxy.

Mirrors the urllib.request usage in methods/llm_agents.py so results
reflect the real call path used by the project.
"""
from __future__ import annotations

import json
import os
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
N_TRIALS = 5
TIMEOUT = 120

KEY_FILE = Path(__file__).resolve().parent / "key.txt"
API_KEY = KEY_FILE.read_text(encoding="utf-8").strip()

PAYLOAD = {
    "model": MODEL,
    "temperature": 0.0,
    "max_tokens": 32,
    "messages": [
        {"role": "system", "content": "Reply with the single word: pong"},
        {"role": "user", "content": "ping"},
    ],
}


def build_opener(use_proxy: bool, proxy_url: str | None):
    if use_proxy and proxy_url:
        handler = urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
    else:
        # explicit empty dict disables auto-pickup of env proxies
        handler = urllib.request.ProxyHandler({})
    return urllib.request.build_opener(handler)


def one_call(opener) -> tuple[float, int, str]:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(PAYLOAD).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with opener.open(req, timeout=TIMEOUT) as resp:
            body = resp.read()
        elapsed = time.perf_counter() - t0
        data = json.loads(body.decode("utf-8"))
        content = data["choices"][0]["message"]["content"].strip()
        return elapsed, len(body), content
    except urllib.error.URLError as exc:
        elapsed = time.perf_counter() - t0
        return elapsed, -1, f"ERROR: {exc}"
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        return elapsed, -1, f"ERROR: {exc!r}"


def dns_probe() -> float:
    t0 = time.perf_counter()
    try:
        socket.getaddrinfo("api.deepseek.com", 443)
    except Exception as exc:  # noqa: BLE001
        print(f"  DNS resolution failed: {exc}")
    return time.perf_counter() - t0


def bench(label: str, use_proxy: bool, proxy_url: str | None) -> list[float]:
    print(f"\n=== {label} ===")
    print(f"  proxy = {proxy_url if use_proxy else '(direct, no proxy)'}")
    print(f"  DNS resolve api.deepseek.com: {dns_probe()*1000:.1f} ms")
    opener = build_opener(use_proxy, proxy_url)
    times: list[float] = []
    for i in range(N_TRIALS):
        elapsed, size, content = one_call(opener)
        marker = "OK" if size > 0 else "FAIL"
        snippet = content if size > 0 else content[:120]
        print(f"  [{i+1}/{N_TRIALS}] {marker} {elapsed:6.2f}s  bytes={size}  reply={snippet!r}")
        times.append(elapsed)
    ok_times = [t for t, _ in zip(times, range(len(times)))]  # keep all; failures shown above
    return ok_times


def summarize(label: str, times: list[float]) -> None:
    if not times:
        print(f"{label}: no data")
        return
    print(
        f"{label}: n={len(times)}  "
        f"mean={statistics.mean(times):.2f}s  "
        f"median={statistics.median(times):.2f}s  "
        f"min={min(times):.2f}s  max={max(times):.2f}s"
    )


def main() -> int:
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    print(f"Python: {sys.version.split()[0]}  Trials per mode: {N_TRIALS}")
    print(f"Detected env proxy: {proxy_url!r}")

    with_proxy = bench("WITH PROXY", use_proxy=True, proxy_url=proxy_url)
    without_proxy = bench("WITHOUT PROXY (direct)", use_proxy=False, proxy_url=None)

    print("\n--- Summary ---")
    summarize("WITH PROXY   ", with_proxy)
    summarize("WITHOUT PROXY", without_proxy)

    if with_proxy and without_proxy:
        wp = statistics.mean(with_proxy)
        np_ = statistics.mean(without_proxy)
        if np_ > 0:
            ratio = wp / np_
            faster = "WITHOUT proxy is faster" if np_ < wp else "WITH proxy is faster"
            print(f"\n{faster}. Proxy/Direct mean ratio = {ratio:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
