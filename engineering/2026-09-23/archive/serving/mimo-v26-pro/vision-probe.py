#!/usr/bin/env python3
"""vision-probe.py [port] -- narrow down why MiMo's vision path returns '?' repeated.

The smoke test sends a valid PNG and gets back 50 identical '?' tokens, which is what a model does when it is handed
embeddings that carry no information -- the same signature the empty mask-token embedding produced in the draft path.
That leaves several candidates, and they are distinguishable by experiment rather than by reading code:

  size        the ViT wants image_size 560, patch 16, spatial_merge 2, and declares image_min_pixels 3136. An image
              below the minimum, or one whose dimensions do not survive patching, could produce a degenerate token grid.
  content     a flat colour field has almost no spatial structure; a textured image exercises the encoder properly.
  text-only   the SAME request without the image is the control. If text-only also degenerates, the problem is the chat
              template or the sampler, not the vision tower at all -- worth one request before blaming the encoder.
  thinking    the smoke disables thinking for this test; MiMo is a reasoning model and behaves differently without it.

Prints the first 120 characters of each answer so the failure mode is visible rather than just pass/fail.
"""
import base64, json, struct, sys, urllib.request, zlib

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18190


def png(w, h, kind="square"):
    rows = []
    for y in range(h):
        row = bytearray(b"\x22\x33\x44" * w)
        if kind == "square":
            if h // 4 < y < 3 * h // 4:
                row[(w // 4) * 3:(3 * w // 4) * 3] = b"\xdd\x11\x11" * (w // 2)
        else:  # textured: diagonal bands, so every patch differs from its neighbours
            for x in range(w):
                v = (x * 7 + y * 13) % 256
                row[x * 3:x * 3 + 3] = bytes([v, (v * 3) % 256, 255 - v])
        rows.append(b"\x00" + bytes(row))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def chat(content, think, n=40):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900))
    except Exception as e:
        return f"<request failed: {e}>"
    return (r["choices"][0]["message"].get("content") or "").strip()


def img_content(w, h, kind, q):
    return [{"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + base64.b64encode(png(w, h, kind)).decode()}},
            {"type": "text", "text": q}]


Q = "What colour is the large square in this image? Answer with one word."
cases = [
    ("text only, no image (CONTROL)", "What colour is a ripe tomato? Answer with one word.", False),
    ("text only, thinking on (CONTROL)", "What colour is a ripe tomato? Answer with one word.", True),
    ("256x256 flat square, no think", img_content(256, 256, "square", Q), False),
    ("256x256 flat square, thinking", img_content(256, 256, "square", Q), True),
    ("560x560 flat square (native size)", img_content(560, 560, "square", Q), False),
    ("560x560 textured", img_content(560, 560, "texture", "Describe this image in one sentence."), False),
    ("64x64 (below min pixels 3136)", img_content(64, 64, "square", Q), False),
]

print(f"{'case':38s} answer")
for label, content, think in cases:
    a = chat(content, think)
    flat = a.replace("\n", " ")[:120]
    print(f"{label:38s} {flat!r}", flush=True)

print()
print("If the two CONTROL rows answer sensibly and every image row degenerates, the vision tower or its projector is")
print("the problem. If the controls also degenerate, look at the chat template and sampling first.")
