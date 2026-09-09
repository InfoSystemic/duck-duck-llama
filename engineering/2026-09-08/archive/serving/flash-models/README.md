# Unsloth Flash models on smeagol

This host has CPU-only, NUMA-local deployments for:

- Qwen3.8-Flash-Next `UD-Q2_K_XL`, revision `83cadfda58d30be06c110518208d1bb918b33f10`
- GLM-5.3-Flash `UD-IQ3_XXS`, revision `ac47690c15c8703615ab7d9c1ef2293d45372757`

The model SSD disappeared from the controller on 2026-08-27. Until that disk
is repaired and checked offline, the verified weights live in `/dev/shm` and
are therefore lost on reboot. Do not access or write the stale `/models` mount.

After a reboot, restage both exact releases (about 201 GB total):

```bash
/home/user/.local/bin/stage-flash-models-to-ram.sh
```

Then start the guarded user services:

```bash
systemctl --user start qwen38-flash-next.service glm53-flash.service
```

The services deliberately are not enabled at boot while their weights are
volatile. The endpoints listen only on localhost:

- Qwen: `http://127.0.0.1:5820/v1`, alias `qwen38-flash-next`
- GLM: `http://127.0.0.1:5830/v1`, alias `glm53-flash`

Both launchers require every shard and projector before starting. They also
reject the `/models` fallback when its backing block device is absent.
GLM defaults to 64K context so it can coexist with Qwen; set `GLM53_CTX_SIZE`
explicitly when testing a larger context window.

The old `glm-sr950.service` boot unit is disabled until its storage is repaired.
The existing Qwen3.8-27B service was left unchanged.
