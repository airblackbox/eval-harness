# eval-harness

Replay episodes, score results, detect regressions for AI agent runs. Part of [AIR Blackbox](https://github.com/airblackbox).

## How It Fits Together

```
Agent / App
    |
    v
AIR Blackbox Gateway ──► records each LLM call
    |
    v
Episode Store ──► groups calls into episodes
    |
    v
Eval Harness ◄── YOU ARE HERE
    |
    ├──► Replay runner (dry or live replay through gateway)
    ├──► Scoring engine (correctness, cost, latency, tools, safety)
    ├──► Regression detector (alerts when scores cross thresholds)
    ├──► Report generator (JSON + Markdown eval reports)
    └──► Nightly scheduler (cron-compatible sweep runner)
```

## Quick Start

```bash
pip install -r requirements.txt

# Check connection to episode store
eval-harness status --store-url http://localhost:8100

# Run an eval
eval-harness run --store-url http://localhost:8100 --limit 20

# Run with filters
eval-harness run --agent-id my-agent --model gpt-4 --output results.json

# Generate report from saved results
eval-harness report results.json
```

## Scoring Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Correctness | 40% | Did the agent reach the same result as baseline? |
| Cost Delta | 20% | Percentage change in token spend |
| Tool Match | 20% | Same tools called in same order? |
| Latency Delta | 10% | Percentage change in duration |
| Safety | 10% | Policy violations detected? |

## Regression Alerts

Alerts fire when scores cross configurable thresholds:

- **Cost** > 20% increase → WARNING; > 40% → CRITICAL
- **Latency** > 30% increase → WARNING; > 60% → CRITICAL
- **Correctness** < 0.5 → CRITICAL
- **Safety** < 0.8 → CRITICAL
- **Overall score** < 0.5 → CRITICAL

## Nightly Runs

```bash
# Run via cron
python -m pkg.scheduler.nightly config.json

# Or with defaults
python -m pkg.scheduler.nightly
```

## Testing

```bash
pytest -v
```

## Roadmap

- [x] Eval schema & scoring models (EV-1)
- [x] Episode fetcher client (EV-2)
- [x] Replay runner — dry + live modes (EV-3)
- [x] Scoring engine — 5 dimensions (EV-4)
- [x] Regression detector with alerts (EV-5)
- [x] CLI runner with Rich output (EV-6)
- [x] Report generator — JSON + Markdown (EV-7)
- [x] Nightly scheduler integration (EV-8)
- [ ] → **Next:** Policy Engine consumes eval scores for autonomy decisions

## License

Apache-2.0