# Puzzle Decoder Race

A solver that reassembles the puzzle in close to one round trip, by keeping a fixed window of requests in flight and stopping as soon as the collected replies statistically rule out a missing fragment.

## How to run

Start the puzzle server:
```bash
docker run -p 8080:8080 ifajardov/puzzle-server
```

Install the one dependency and run the solver:
```bash
pip install -r requirements.txt
python main.py
```

The assembled message goes to stdout; a diagnostic report goes to stderr. To keep only the message:
```bash
python main.py 2>/dev/null
```

The server URL can be overridden with `PUZZLE_SERVER_URL`. Tests run offline against a fake transport, so Docker is not needed for them:
```bash
pip install pytest
pytest -q
```

## Strategy

### Correctness: a stopping rule
The total number of fragments is completely unknown prior to execution. To handle any puzzle size dynamically, the solver never relies on a hardcoded target; instead, completion is inferred in three steps based on the number of distinct fragments observed so far ($k$):   

1. **Contiguity.** Observed indices must form an unbroken sequence starting at 0 (0..k-1). Any gap rules out completion immediately.   
2. **Dynamic Confidence Bound.** If an unseen fragment existed, the worst-case scenario is a puzzle of size $k + 1$. The probability of failing to draw that missing piece in $n$ total replies is $(k / (k + 1))^n$. Requiring this bound to stay below $\delta = 10^{-3}$ yields a dynamic minimum sample threshold $n \ge \left\lceil \frac{\ln(\delta)}{\ln(k / (k + 1))} \right\rceil$ that automatically scales as new pieces $k$ are discovered.   
3. **Good-Turing Check.** The statistical bound assumes roughly uniform sampling. Requiring zero fragments seen only once ($f_1 = 0$) ensures robustness even under heavily skewed distributions: a piece that has only appeared once indicates that additional missing fragments may still be surfacing.   

### Performance: a moving window
Wall-clock execution time depends on sequential round-trip latency, not total request volume. Sequential fetching suffers from the Coupon Collector's Problem ($k \cdot H_k$ probes on average), resulting in cumulative delays that grow non-linearly with puzzle size.   

Conversely, firing hundreds of unthrottled requests saturates the server's connection queue and degrades response times. The solver instead maintains a fixed `WINDOW` of active requests in flight, dispatching replacements immediately as responses land to maintain maximum network throughput without overloading the connection pool.   

Leading ID blocks are requested with multiple replicas (`REPLICAS`) to pick the fastest response attempt per fragment. Replies are processed as they arrive, and all remaining tasks are canceled the moment completion is verified.   

### Creativity: hedging against an unknown id mapping
The mapping strategy from request ID to fragment index is unspecified. The probing schedule combines consecutive IDs, strided IDs (using a large prime to prevent bucket collisions), and seeded pseudo-random IDs to guarantee coverage regardless of internal server indexing.   

Redundancy replaces retries: because each ID space is covered redundantly, dropped or failed HTTP requests are simply ignored without introducing blocking retry logic or backoff penalties.   

### Code quality
One dependency, one module, and the stopping rule expressed as small, independently testable functions. The test suite exercises the real HTTP client and concurrency model using a custom `AsyncBaseTransport`, covering arbitrary puzzle sizes (such as 1, 10, and 30 fragments), skewed distributions, 20% server failure rates, index gaps, and latency bounds.   

Key architectural choices:
* `httpx.Limits` is explicitly configured to match `WINDOW` to prevent internal connection pool queueing.   
* The base URL defaults to an explicit loopback address (`127.0.0.1`) to bypass DNS resolution overhead.   
* Process execution timing begins before module imports to capture total runtime accurately.   

## Bonus: under one second

```text
hello world quick brown fox jumps over lazy dog you have to call all request at same time if you want to see the puzzle fragments fast enough

probes launched     : 316
replies used        : 197
failed probes       : 0
distinct fragments  : 28
unobserved mass (GT): 0.0000
time to first reply : 0.232 s
time in flight      : 0.530 s
total wall clock    : 0.763 s
complete            : yes
```