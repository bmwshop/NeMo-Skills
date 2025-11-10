import sys

in_path = "/lustre/fsw/portfolios/llmservice/users/drekesh/DATA/lc/lcrqagen_v2/output.jsonl"
out_path = "/lustre/fsw/portfolios/llmservice/users/drekesh/DATA/lc/lcrqagen_v2/output_safe.jsonl"

CHUNK = 1024 * 1024  # 1 MB chunks

count = 0
ccount=0
with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
    buf = b""
    while True:
        chunk = fin.read(CHUNK)
        ccount += 1
        if not chunk:
            break
        buf += chunk
        # Split safely on '\n' if present
        print(f"ccount: {ccount}")
        while True:
            nl = buf.find(b"\n")
            if nl == -1:
                break
            line, buf = buf[:nl], buf[nl + 1:]
            # Decode line safely and drop undecodable bytes
            try:
                clean = line.decode("utf-8", errors="ignore").encode("utf-8")
            except Exception:
                clean = b""
            if clean:
                fout.write(clean + b"\n")
                count += 1
                if count % 1 == 0:
                    print("Processed", count, "lines")
    # Write final tail if missing newline
    if buf:
        clean = buf.decode("utf-8", errors="ignore").encode("utf-8")
        fout.write(clean + b"\n")

print("✅ Finished cleaning:", count, "lines written")

