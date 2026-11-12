# Benchmark results

| dataset | mode | model | judge | searchers | n | accuracy | macro-F1 | FEVER score | gold page cited | fake citations | $ / claim | s / claim | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fever | team | qwen.qwen3-235b-a22b-2507 | llm | wiki,kb,papers | 300 | 69.3% | 0.663 | 57.0% | 90.0% | 1.9% | 0.0015 | 12.2 | 0 |
| fever | single | qwen.qwen3-235b-a22b-2507 | llm | wiki,kb,papers | 300 | 66.7% | 0.626 | 9.0% | 57.5% | 22.1% | 0.0021 | 8.5 | 0 |
| arafacts | single | qwen.qwen3-235b-a22b-2507 | llm | wiki,kb,papers | 125 | 30.4% | 0.308 | - | - | 5.6% | 0.0037 | 15.5 | 0 |
| arafacts | team | qwen.qwen3-235b-a22b-2507 | llm | wiki,kb,papers | 125 | 28.0% | 0.284 | - | - | 6.2% | 0.0025 | 18.2 | 0 |

`?` after a cost = some calls had no known price. FEVER score = label correct and a complete gold evidence set cited (KB citations only).
