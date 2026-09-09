#!/usr/bin/env python3
"""Build isolated packed/padded completion counters; never edit Full's build."""
import difflib
import hashlib
from pathlib import Path
import json
import shlex


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(engine, destination, run):
    engine, destination = Path(engine), Path(destination)
    destination.mkdir(exist_ok=False)
    build_dir = engine / "build-dev2"
    source_path = engine / "ggml/src/ggml-backend-meta.cpp"
    source = source_path.read_text()
    old = "    alignas(64) std::atomic<int> done_threads[GGML_META_FUSED_MAX_DEVS];"
    new = """    struct alignas(GGML_PRIVATE_PAD_COUNTERS ? 64 : alignof(std::atomic<int>)) completion_counter {
        std::atomic<int> value;
    };
    static_assert(sizeof(completion_counter) == (GGML_PRIVATE_PAD_COUNTERS ? 64 : sizeof(std::atomic<int>)));
    alignas(64) completion_counter done_threads[GGML_META_FUSED_MAX_DEVS];"""
    assert source.count(old) == 1 and source.count("done_threads[j].") == 3
    changed = source.replace(old, new).replace("done_threads[j].", "done_threads[j].value.")
    private_source = destination / "ggml-backend-meta.cpp"
    private_source.write_text(changed)
    (destination / "counter-layout.patch").write_text("".join(difflib.unified_diff(
        source.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="a/ggml/src/ggml-backend-meta.cpp", tofile="b/ggml/src/ggml-backend-meta.cpp")))

    commands_path = build_dir / "compile_commands.json"
    entries = [entry for entry in json.loads(commands_path.read_text()) if entry["file"] == str(source_path)]
    assert len(entries) == 1
    entry = entries[0]
    cwd = Path(entry["directory"])
    compile_command = shlex.split(entry["command"])
    # compile_commands contains escaped string-literal definitions. subprocess
    # receives literal quotes directly, without a shell consuming the escapes.
    compile_command = [arg.replace('\\"', '"') if arg.startswith(("-DGGML_COMMIT=", "-DGGML_VERSION=")) else arg
                       for arg in compile_command]
    output_index = compile_command.index("-o") + 1
    old_object = (cwd / compile_command[output_index]).resolve()
    assert old_object.name == "ggml-backend-meta.cpp.o"
    assert compile_command.count(str(source_path)) == 1

    link_path = cwd / "CMakeFiles/ggml-base.dir/link.txt"
    link_command = shlex.split(link_path.read_text())
    link_output_index = link_command.index("-o") + 1
    library_name = Path(link_command[link_output_index]).name
    assert library_name.startswith("libggml-base.so.")
    objects = [str((cwd / arg).resolve()) for arg in link_command if arg.endswith(".o")]
    assert objects.count(str(old_object)) == 1
    originals = [source_path, commands_path, link_path] + [Path(p) for p in objects]
    for folder in (engine / "ggml/src", engine / "ggml/include"):
        originals += list(folder.glob("*.h")) + list(folder.glob("*.hpp"))
    info = dict(input_sha256={str(p): digest(p) for p in originals}, variants={},
                private_source_sha256=digest(private_source),
                scope="Private libggml-base variants; all original source, objects, and libraries remain read-only.")
    (destination / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    for mode, padded in (("packed", 0), ("padded", 1)):
        directory = destination / mode
        directory.mkdir()
        obj = directory / "ggml-backend-meta.cpp.o"
        library = directory / library_name
        command = compile_command.copy()
        command[output_index] = str(obj)
        command[command.index(str(source_path))] = str(private_source)
        command.append(f"-DGGML_PRIVATE_PAD_COUNTERS={padded}")
        run(command, cwd, f"counter-{mode}-compile.log")
        link = link_command.copy()
        link[link_output_index] = str(library)
        for index, arg in enumerate(link):
            if arg.endswith(".o"):
                original = (cwd / arg).resolve()
                link[index] = str(obj if original == old_object else original)
        run(link, cwd, f"counter-{mode}-link.log")
        (directory / "libggml-base.so.0").symlink_to(library.name)
        (directory / "libggml-base.so").symlink_to("libggml-base.so.0")
        info["variants"][mode] = dict(directory=str(directory), library=str(library),
                                     library_sha256=digest(library), object_sha256=digest(obj),
                                     compile_command=command, link_command=link,
                                     counter_stride_bytes=64 if padded else 4)
        (destination / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    assert all(digest(p) == checksum for p, checksum in info["input_sha256"].items())
    return info
