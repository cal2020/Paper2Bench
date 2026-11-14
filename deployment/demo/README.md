# Demo deployment

The demo is a static, interactive presentation of Paper2Bench 0.4.0a2. Recorded
synthetic results are labeled; this deployment does not run candidate code,
accept provider credentials, or expose the controller runtime.

Source: `docs/` in [cal2020/Paper2Bench](https://github.com/cal2020/Paper2Bench).
The Site's stable identity and static directory are tracked in
[.openai/hosting.json](.openai/hosting.json). Access remains owner-private.

Build from the repository root:

```bash
python3 scripts/build_demo.py
python3 -m http.server 8765 --bind 127.0.0.1 --directory deployment/demo/dist
```

For publication, build into a separate Site source checkout using `--output`.
Use the Sites source workflow to commit/push that exact build, package it, and
save/deploy the returned source commit. Never copy `.venv`, `runs`, `.env`, model
credentials, Git metadata, or private curator/runtime state into the assets.

Production: https://paper2bench-portable-demo.escape4242.chatgpt.site

The Python CLI/SDK is installed from GitHub and executes on an operator's local
or SSH Docker worker. This static deployment has no compute API or job service.
