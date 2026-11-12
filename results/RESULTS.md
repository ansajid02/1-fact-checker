# Benchmark results

| dataset | mode | model | judge | searchers | n | accuracy | macro-F1 | FEVER score | gold page cited | fake citations | $ / claim | s / claim | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fever | single | qwen.qwen3-235b-a22b-2507 | llm | wiki,kb,papers | 30 | 63.3% | 0.563 | 3.3% | 60.0% | 29.3% | 0.0017 | 9.2 | 0 |
| fever | team | qwen.qwen3-235b-a22b-2507 | llm | wiki,kb,papers | 30 | 56.7% | 0.478 | 40.0% | 85.0% | 0.0% | 0.0017 | 13.8 | 1 |

`?` after a cost = some calls had no known price. FEVER score = label correct and a complete gold evidence set cited (KB citations only).
