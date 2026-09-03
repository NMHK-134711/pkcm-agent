"""Does --gpu-server make a 3200-sim self-play battle cheaper? Measure it.

Three configs, identical seeds (so identical teams), one checkpoint:
  cpu16  workers hold the net on CPU, leaf_batch 16   -- what A/B ran
  gpu16  one CUDA server, leaf_batch 16               -- same search, forward moved
  gpu64  one CUDA server, leaf_batch 64               -- what NEXT_RUN's alternative asks

Measured 2026-09-02 on the personal PC (Core Ultra 7 265KF, RTX 5060 Ti,
19 workers): cpu16 10.57 s/battle, gpu16 7.65 (1.38x), gpu64 7.95 (1.33x).
Notion s13's "forward ~6%" was a single-process cProfile at 800 sims and does
not describe this condition -- the earlier "6% ceiling" claim was wrong.
Usage: python gpu_server_throughput.py <battles> [workers]   (40 10 on the lab)
Run it only on an idle machine; a live training run shares the cores.
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))

def main():
    from pkcm.search.mcts import SearchConfig
    from pkcm.train.samples import SelfPlayConfig
    from pkcm.train.parallel import generate
    battles = int(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 19
    seed = 424242
    configs = [("cpu16", False, 16), ("gpu16", True, 16), ("gpu64", True, 64)]
    rows = []
    for name, gpu, leaf in configs:
        cfg = SelfPlayConfig(
            teams="parties:43", foe_teams="parties",
            checkpoint=str(ROOT / "runs/imitate8/net.pt"), trust=1.0,
            search=SearchConfig(iterations=3200, determinizations=160,
                                leaf_batch=leaf, root_noise=0.25, sample_turns=12))
        print(f"\n== {name}: gpu_server={gpu} leaf_batch={leaf} {battles} battles {workers} workers", flush=True)
        t0 = time.perf_counter(); n = 0; samples = 0
        for batch in generate(cfg, battles, seed=seed, workers=workers, gpu_server=gpu):
            n += 1; samples += len(batch)
        dt = time.perf_counter() - t0
        rows.append(dict(config=name, gpu_server=gpu, leaf_batch=leaf, battles=n,
                         samples=samples, seconds=round(dt, 1),
                         s_per_battle=round(dt / max(n, 1), 2),
                         battles_per_s=round(n / dt, 4)))
        print(f"   {n} battles  {samples} samples  {dt:.0f}s  {dt/max(n,1):.1f} s/battle", flush=True)
    base = rows[0]["s_per_battle"]
    print("\n== summary (3200 sims, imitate8, parties:43 vs parties, same seeds) ==")
    print(f"{'config':7} {'s/battle':>9} {'vs cpu16':>9} {'samples':>8}")
    for r in rows:
        print(f"{r['config']:7} {r['s_per_battle']:>9.2f} {base/r['s_per_battle']:>8.2f}x {r['samples']:>8}")
    (ROOT / "runs/gpu_server_throughput.json").write_text(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
