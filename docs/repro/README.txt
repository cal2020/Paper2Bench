Paper2Bench optional reproduction core: recorded public evidence

These files select evidence from the archived local v0.2 core. The historical
verification receipt records 642 pytest cases plus 93 subtests; those counts
are not the current canonical repository suite. Optimization and seed-study
data are synthetic software fixtures, not a published-paper reproduction.
The separate Adam case study remains at ../results/adam-counterexample.json.

evidence-v0.2.zip contains selected reports, public synthetic data/predictions,
and their checksums. It is NOT a full runtime replay bundle, a wheel, a live
execution service, or proof that the current namespaced code ran these jobs.
No credentials, worker identities, state databases, or private runtime logs are
included. Hashes establish integrity; scientific fidelity remains unverified.

Install the canonical repository, Python 3.11+:
  git clone https://github.com/cal2020/Paper2Bench.git
  cd Paper2Bench
  python -m pip install -e '.[repro]'
  paper2bench-repro --help
  paper2bench-repro --state .paper2bench-repro demo
  paper2bench-repro --state .paper2bench-repro demo --task knapsack

These trusted builtin demos require neither Docker nor model credentials.
Python candidate runs with --python require a local Docker daemon and staged
Python image; see docs/REPRODUCIBILITY.md for placement, images, limits, and
provider setup. Fresh demos create NEW runs distinct from this recorded evidence.
The optional CLI is paper2bench-repro. The original paper2bench CLI and
canonical distribution identity remain unchanged. Archived standalone v0.2
paper2bench wheels are deliberately not redistributed by this export.

Regenerate from the original local artifacts, then commit the public outputs:
  python scripts/export_repro_demo.py --source-dist /path/to/repro-core/dist
Verify all generated files and ZIP members without the original local state:
  python scripts/export_repro_demo.py --verify-only
Build the committed static demo using the existing provenance-preserving flow:
  python scripts/build_demo.py --release

The static build recursively includes these JSON/TXT/ZIP assets and binds them
to build-provenance.json and asset-manifest.json. It makes no provider calls.
