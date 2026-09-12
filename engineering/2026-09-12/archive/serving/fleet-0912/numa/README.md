# NUMA placement repair for staged GLM Flash weights

Status: built and validated only on an owned 4 MiB tmpfs fixture. No model pages have been migrated by this utility. Model-scale application remains unverified and requires review of the resulting placement and memory headroom. Files are frozen for the repository snapshot requested by the user.

The utility reads the six exact GLM-5.3-Flash UD-Q4_K_XL shard paths on this machine. It opens them O_RDONLY, maps them PROT_READ/MAP_SHARED, and uses Linux move_pages with pid=0 and flags=0. It never writes weights, copies model files, uses MOVE_ALL, requests root privileges, or modifies persistent memory policy. Migration changes the NUMA placement of existing tmpfs pages.

```
bash build.sh
./rebalance-tmpfs --inspect
./rebalance-tmpfs --inventory --json new-inventory.json
```

Default inspection queries 4096 evenly spaced pages per file. `--inventory` queries every page. Both modes fault selected pages into this process in bounded batches without requesting migration. Progress is printed to stderr about every 15 seconds; stdout contains the final JSON report. Run in the host namespace, as restricted sandboxes can block move_pages.

After review, the explicit mutation mode is:

```
./rebalance-tmpfs --apply --max-move-gib 64 --max-seconds 1200 --json new-apply-report.json
```

Apply inventories all six files before moving any page. It computes equal per-file four-node targets and moves only pages from nodes above their target to nodes below it. The full initial plan must fit the migration budget. Default limits are 4096 pages per syscall batch, 64 GiB of page moves, and 1200 seconds. There is one bounded migration pass per file, followed by a full placement re-query; unsuccessful moves or external placement changes can leave an uneven file and produce a nonzero exit. There is no automatic retry. Completed moves remain in place on interruption or deadline expiry.

Before mapping, the utility checks regular-file type, exact known shard sizes, effective-UID ownership, tmpfs filesystem type and the host's 4096-byte page size. It refuses root execution. Before migration and approximately once per second it checks device/inode/size/mtime/ctime stability and refuses to proceed if any process named llama-server maps a target inode. Another model using different files does not trigger that guard. A newly appearing or renamed mapper is still subject to the kernel's ordinary shared-page migration restrictions; no MOVE_ALL bypass is used. These checks do not reserve memory or prevent another controller from launching a model.

The report records sampled or complete node counts, syscall and per-page errors, attempted and successful migrations, bytes moved, and before/after SHA-256 of up to 1024 evenly spaced pages per file including offsets. This sample digest is not a full-model integrity proof. Report files must be new paths: the utility will not overwrite or follow an existing report path.

`fixture-verification.json` records the bounded test: a new 4 MiB tmpfs file began with 1024 pages on node 0. The utility moved 3 MiB and verified final counts of 256 pages on each node. An external sha256sum of the entire fixture was identical before and after. The temporary fixture was then removed. Raw utility evidence is in `fixture-inspect.json` and `fixture-apply.json`.

Build dependencies are a C++17 compiler, OpenSSL libcrypto/headers, and the existing nlohmann JSON header under engines/llama.cpp-sr950-glm/vendor. The build script deliberately references this machine's existing engine checkout. The syscall behavior was checked against the locally installed move_pages(2) manual and a real fixture; large model migration has not been executed.
