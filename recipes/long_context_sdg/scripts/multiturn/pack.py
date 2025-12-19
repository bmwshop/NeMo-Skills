#!/usr/bin/env python
import os
import sys
import json
import random
import logging
import threading
import queue
import argparse
from tqdm import tqdm

try:
    from tokenizer.tiktokenizer import TiktokenTokenizer
except ImportError:
    print('tiktoken tokenizer not available')
try:
    from transformers import AutoTokenizer
except ImportError:
    print('transformers library not available for HuggingFace tokenizers')

# adapted from /lustre/fsw/portfolios/llmservice/users/skriman/lgen/scripts/create_extended_dataset7.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr
)

META_FILENAME = "metadata.json"
MAX_LINE_SIZE = 10 * 1024 * 1024
QUEUE_PUT_TIMEOUT = 5

conv1_fhs = None
conv2_fhs = None
metadata = None
output_folder = None

def setup_tokenizer(tokenizer_path):
    """Setup the tokenizer based on the provided path or HuggingFace URL"""

    # Check if it's a HuggingFace URL (contains '/' and doesn't exist as local file)
    # if '/' in tokenizer_path and not os.path.exists(tokenizer_path):
    if not tokenizer_path.endswith('.json'):
        try:
            print(f"Loading HuggingFace tokenizer: {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace tokenizer '{tokenizer_path}': {e}")
    else:
        # Treat as tiktoken tokenizer
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        try:
            from tokenizer.tiktokenizer import TiktokenTokenizer
            print(f"Loading local tiktoken tokenizer: {tokenizer_path}")
            tokenizer = TiktokenTokenizer(vocab_file=tokenizer_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load local tokenizer '{tokenizer_path}': {e}")
    return tokenizer

def tokenize_text(text, tokenizer):
    if tokenizer is None:
        raise RuntimeError("Tokenizer not initialized. Call setup_tokenizer first.")

    # Handle different tokenizer types
    if hasattr(tokenizer, 'text_to_tokens'):
        # TiktokenTokenizer
        return tokenizer.text_to_tokens(text)
    else:
        # HuggingFace AutoTokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return tokens


# ---------- helpers for metadata loading/estimation ----------
def _try_load_metadata(folder):
    path = os.path.join(folder, META_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _count_lines(path):
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def _estimate_metadata(
    folder,
    samples_per_file,
    T_final,
    main_paths,
    conv1_paths,
    conv2_paths,
    add_dep1,
    add_dep2,
    main_props,
):
    existing_outputs = [
        f for f in os.listdir(folder)
        if f.startswith("output_") and f.endswith(".jsonl")
    ]
    num_out = len(existing_outputs)
    est_samples = num_out * samples_per_file

    # ---- stats skeleton ----
    buckets = [{"count": 0, "min_tokens": None, "max_tokens": None} for _ in range(10)]
    lb = T_final - 5000
    for i in range(10):
        buckets[i]["range"] = f"{lb + i*2000}-{lb + (i+1)*2000}"
    if est_samples:
        base, rem = divmod(est_samples, 10)
        for i in range(10):
            buckets[i]["count"] = base + (1 if i < rem else 0)

    meta = {
        "main_files_offsets": [],
        "main_files_cycles": [],
        "conv1_indices": [],
        "conv1_cycles": [],
        "conv2_indices": [],
        "conv2_cycles": [],
        "stats": {
            "total_conversations": est_samples,
            "total_tokens": est_samples * T_final,
            "min_tokens": None,
            "max_tokens": T_final if est_samples else None,
            "mean_tokens": T_final if est_samples else 0,
            "buckets": buckets,
        },
        "dependency_stats": {},
        "current_output_file_index": 1,
    }

    # ---- estimate per‑file positions ----
    main_consumed = est_samples
    if main_props is None:
        main_props = [1.0 / len(main_paths)] * len(main_paths)
    total_prop = sum(main_props)
    main_props = [p / total_prop for p in main_props]

    for pth, w in zip(main_paths, main_props):
        tot = _count_lines(pth) or 1
        used = int(main_consumed * w)
        cycles, offset = divmod(used, tot)
        meta["main_files_offsets"].append(offset)
        meta["main_files_cycles"].append(cycles)

    dep1_consumed = est_samples * add_dep1
    dep2_consumed = est_samples * add_dep2

    for pth in conv1_paths:
        tot = _count_lines(pth) or 1
        used = dep1_consumed // len(conv1_paths)
        cycles, offset = divmod(used, tot)
        meta["conv1_indices"].append(offset)
        meta["conv1_cycles"].append(cycles)

    for pth in conv2_paths:
        tot = _count_lines(pth) or 1
        used = dep2_consumed // len(conv2_paths)
        cycles, offset = divmod(used, tot)
        meta["conv2_indices"].append(offset)
        meta["conv2_cycles"].append(cycles)

    if existing_outputs:
        try:
            idxs = [int(f.split("_")[1].split(".")[0]) for f in existing_outputs]
            meta["current_output_file_index"] = max(idxs) + 1
        except ValueError:
            meta["current_output_file_index"] = 1
    return meta


def save_metadata(folder, meta):
    path = os.path.join(folder, META_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def open_output_file(folder, idx):
    return open(os.path.join(folder, f"output_{idx:05d}.jsonl"), "w", encoding="utf-8")


# ---------- ensure list helper ----------
def ensure_list(meta, key, desired_len, fill_value=0):
    if key not in meta or not isinstance(meta[key], list):
        meta[key] = [fill_value] * desired_len
    else:
        if len(meta[key]) < desired_len:
            meta[key].extend([fill_value] * (desired_len - len(meta[key])))
        elif len(meta[key]) > desired_len:
            meta[key] = meta[key][:desired_len]


# ---------- utility functions (unchanged) ----------
def sanitize_conversation(conv):
    out = []
    for turn in conv:
        new_t = dict(turn)
        val = new_t.get("value", "")
        if not isinstance(val, str):
            try:
                new_t["value"] = json.dumps(val, ensure_ascii=False)
            except Exception:
                new_t["value"] = str(val)
        out.append(new_t)
    return out

def sanitize_conversation_oai(conv):
    out = []
    for turn in conv:
        new_t = dict(turn)
        val = new_t.get("content", "")
        if "reasoning_content" in new_t:
            new_t.pop("reasoning_content")

        if not isinstance(val, str):
            try:
                new_t["content"] = json.dumps(val, ensure_ascii=False)
            except Exception:
                new_t["content"] = str(val)
        out.append(new_t)
    return out

def count_tokens(conv, sp):
    txts = []
    for turn in conv:
        val = turn.get("value", "")
        if not isinstance(val, str):
            try:
                val = json.dumps(val, ensure_ascii=False)
            except Exception:
                val = str(val)
        txts.append(val)
    return len(tokenize_text(" ".join(txts), sp))

def count_turn_list_tokens_oai(turns, sp):
    txts = []
    for t in turns:
        val = t.get("content", "")
        if t.get("role") == "system":
            continue
        txts.append(val)
    # return len(sp.encode(" ".join(txts)))
    return len(tokenize_text(" ".join(txts), sp))


def count_turn_list_tokens(turns, sp):
    txts = []
    for t in turns:
        val = t.get("value", "")
        if not isinstance(val, str):
            try:
                val = json.dumps(val, ensure_ascii=False)
            except Exception:
                val = str(val)
        txts.append(val)
    # return len(sp.encode(" ".join(txts)))
    return len(tokenize_text(" ".join(txts), sp))

def count_turn_list_tokens_oai(turns, sp):
    txts = []
    for t in turns:
        val = t.get("content", "")
        if t.get("role") == "system":
            continue
        txts.append(val)
    # return len(sp.encode(" ".join(txts)))
    return len(tokenize_text(" ".join(txts), sp))


def merge_conversations(conv1, conv2):
    if conv1 and conv2 and conv1[-1]["from"] == conv2[0]["from"]:
        dummy = {
            "from": "Assistant" if conv1[-1]["from"] == "User" else "User",
            "value": "",
        }
        return conv1 + [dummy] + conv2
    return conv1 + conv2

def merge_conversations_oai(conv1, conv2):
    if conv1 and conv2 and conv1[-1]["role"] == conv2[0]["role"]:
        dummy = {
            "role": "assistant" if conv1[-1]["role"] == "user" else "user",
            "content": "",
        }
        return conv1 + [dummy] + conv2
    return conv1 + conv2

def insert_block(conv, block, boundary):
    if not isinstance(block, list):
        block = [block]
    nb = boundary
    while nb > 0 and nb <= len(conv) and conv[nb - 1].get("from") != "Assistant":
        nb += 1
    if nb > len(conv):
        nb = len(conv)
    return conv[:nb] + block + conv[nb:]

def insert_block_hybrid(conv, block, boundary):
    # the conv is in the new format: messages
    # the block is in the old format
    if not isinstance(block, list):
        block = [block]
    # convert the block to the new format
    block = [{"role": t.get("from").lower(), "content": t.get("value")} for t in block]

    nb = boundary
    while nb > 0 and nb <= len(conv) and conv[nb - 1].get("role").lower() != "assistant":
        nb += 1
    if nb > len(conv):
        nb = len(conv)
    return conv[:nb] + block + conv[nb:]


def is_invalid_ref_turn(tp):
    for turn in tp:
        v = turn.get("value", "").lower()
        if ("q:" in v and "a:" in v) or ("user" in v and "assistant" in v):
            return True
    return False


def is_invalid_segment(seg):
    for turn in seg:
        v = turn.get("value", "")
        if "User" in v and "Assistant" in v:
            return True
    return False


def update_stats(meta, tok_count, T_final):
    st = meta["stats"]
    st["total_conversations"] += 1
    st["total_tokens"] += tok_count
    if st["min_tokens"] is None or tok_count < st["min_tokens"]:
        st["min_tokens"] = tok_count
    if st["max_tokens"] is None or tok_count > st["max_tokens"]:
        st["max_tokens"] = tok_count

    lb = T_final - 5000
    bbd = [lb + (i + 1) * 2000 for i in range(10)]
    for i in range(10):
        st["buckets"][i]["range"] = f"{lb + i*2000}-{lb + (i+1)*2000}"
    idx = 0
    for i, bound in enumerate(bbd):
        if tok_count < bound:
            idx = i
            break
    else:
        idx = 9
    buck = st["buckets"][idx]
    buck["count"] += 1
    if buck["min_tokens"] is None or tok_count < buck["min_tokens"]:
        buck["min_tokens"] = tok_count
    if buck["max_tokens"] is None or tok_count > buck["max_tokens"]:
        buck["max_tokens"] = tok_count


def all_even_boundaries(length):
    return [i for i in range(length + 1) if i % 2 == 0]


# ------------------- MAIN FILE CYCLING LOGIC -------------------
def skip_main_offset(fh, skipcount, meta_lock, file_idx):
    global metadata
    cyc = metadata["main_files_cycles"][file_idx]
    read_so_far = 0
    while read_so_far < skipcount:
        if not fh.readline():
            fh.seek(0)
            cyc += 1
        else:
            read_so_far += 1
    with meta_lock:
        metadata["main_files_cycles"][file_idx] = cyc


def read_line_cycling(fh, file_idx, meta_lock):
    global metadata
    old_cyc = metadata["main_files_cycles"][file_idx]
    old_offset = metadata["main_files_offsets"][file_idx]

    line = fh.readline()
    if not line:
        fh.seek(0)
        metadata["main_files_cycles"][file_idx] = old_cyc + 1
        metadata["main_files_offsets"][file_idx] = 0
        line = fh.readline()
        if not line:
            return None
        metadata["main_files_offsets"][file_idx] += 1
        return line

    metadata["main_files_offsets"][file_idx] = old_offset + 1
    return line


# ------------------- DEP READING -------------------
def get_dep_line_dep1():
    global conv1_fhs, metadata
    fidx = random.randint(0, len(conv1_fhs) - 1)
    metadata["conv1_indices"][fidx] += 1
    line = conv1_fhs[fidx].readline()
    if not line:
        conv1_fhs[fidx].seek(0)
        line = conv1_fhs[fidx].readline()
        metadata["conv1_cycles"][fidx] += 1
    if len(line) > MAX_LINE_SIZE:
        return None
    return line.strip()


def get_dep_line_dep2():
    global conv2_fhs, metadata
    fidx = random.randint(0, len(conv2_fhs) - 1)
    metadata["conv2_indices"][fidx] += 1
    line = conv2_fhs[fidx].readline()
    if not line:
        conv2_fhs[fidx].seek(0)
        line = conv2_fhs[fidx].readline()
        metadata["conv2_cycles"][fidx] += 1
    if len(line) > MAX_LINE_SIZE:
        return None
    return line.strip()


# ---------------- dependency insertion helpers (unchanged) ----------------
def insert_dependency_dep1_partial(sample, sample_token_count, sp, T_final, dep_obj, num_ref_turns):
    # the dep_obj is still in the old format. 
    block_orig = dep_obj.get("original_conversation", {}).get("conversations", [])
    if "user_rewrite" in dep_obj and block_orig:
        rw = dep_obj["user_rewrite"]
        for t in block_orig:
            if t.get("from") == "User":
                t["value"] = rw
                break
    referencing = dep_obj.get("referencing_turns", [])
    referencing = [r for r in referencing if not is_invalid_ref_turn(r)]
    if num_ref_turns is not None and num_ref_turns <= len(referencing):
        referencing = [referencing[i] for i in sorted(random.sample(range(len(referencing)), num_ref_turns))]

    boundaries = sorted(all_even_boundaries(len(sample)))
    if not boundaries:
        return sample, sample_token_count
    earliest = boundaries[0]
    offset = 0
    needed_blk = count_turn_list_tokens(block_orig, sp) if block_orig else 0
    if block_orig and sample_token_count + needed_blk <= T_final:
        sample = insert_block_hybrid(sample, block_orig, earliest + offset)
        sample_token_count += needed_blk
        offset += len(block_orig)
    else:
        block_orig = []

    remain = [b for b in boundaries if b > earliest]
    if not remain or not referencing:
        return sample, sample_token_count
    random.shuffle(referencing)
    random.shuffle(remain)
    for i in range(min(len(referencing), len(remain))):
        needed_ref = count_turn_list_tokens(referencing[i], sp)
        if sample_token_count + needed_ref <= T_final:
            bpos = remain[i]
            sample = insert_block_hybrid(sample, referencing[i], bpos + offset)
            sample_token_count += needed_ref
            offset += len(referencing[i])
        else:
            break
    return sample, sample_token_count


def insert_dependency_dep2_partial(sample, sample_token_count, sp, T_final, dep_obj):
    segs = dep_obj.get("segments", [])
    valid = [s for s in segs if not is_invalid_segment(s)]
    if not valid:
        return sample, sample_token_count
    bounds = all_even_boundaries(len(sample))
    if not bounds:
        return sample, sample_token_count
    needed_count = min(len(valid), len(bounds))
    chosen = sorted(random.sample(bounds, needed_count))
    offset = 0
    for i, seg in enumerate(valid[:needed_count]):
        seg_tok = count_turn_list_tokens(seg, sp)
        if sample_token_count + seg_tok <= T_final:
            bpos = chosen[i]
            sample = insert_block(sample, seg, bpos + offset)
            sample_token_count += seg_tok
            offset += len(seg)
        else:
            break
    return sample, sample_token_count


def insert_dependencies_until_threshold_partial(
    sample,
    sample_token_count,
    sp,
    T_final,
    add_dep1,
    add_dep2,
    dep1_fn,
    dep2_fn,
    num_ref_turns,
):
    total_dep1 = 0
    total_dep2 = 0
    while True:
        for _ in range(add_dep1):
            ln = dep1_fn()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            old_s = sample
            old_ct = sample_token_count
            sample, sample_token_count = insert_dependency_dep1_partial(
                sample,
                sample_token_count,
                sp,
                T_final,
                obj,
                num_ref_turns,
            )
            if sample is old_s and sample_token_count == old_ct:
                return sample, sample_token_count, total_dep1, total_dep2
            total_dep1 += 1

        for _ in range(add_dep2):
            ln2 = dep2_fn()
            if not ln2:
                continue
            try:
                obj2 = json.loads(ln2)
            except Exception:
                continue
            old_s = sample
            old_ct = sample_token_count
            sample, sample_token_count = insert_dependency_dep2_partial(
                sample, sample_token_count, sp, T_final, obj2
            )
            if sample is old_s and sample_token_count == old_ct:
                return sample, sample_token_count, total_dep1, total_dep2
            total_dep2 += 1
    return sample, sample_token_count, total_dep1, total_dep2


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Main files cycling with proportion-based picking, metadata cycles etc."
    )
    parser.add_argument("--main_files", required=True)
    parser.add_argument("--conv1_files", required=True)
    parser.add_argument("--conv2_files", required=False, default=None)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--tokenizer_model", required=True)
    parser.add_argument("--T_final", type=int, required=True)
    parser.add_argument("--T_main", type=int, required=True)
    parser.add_argument("--add_long_dep_1", type=int, required=True)
    parser.add_argument("--add_long_dep_2", type=int, required=False, default=0)
    parser.add_argument("--samples_per_file", type=int, required=True)
    parser.add_argument("--max_main_lines", type=int, required=True)
    parser.add_argument("--mode", choices=["discard", "allow_overflow"], required=True)
    parser.add_argument("--num_ref_turns", type=int, default=None)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--num_worker_threads", type=int, default=16)
    parser.add_argument("--main_files_proportions", default=None)

    args = parser.parse_args()

    main_paths = args.main_files.split(",")
    conv1_paths = args.conv1_files.split(",")
    conv2_paths = args.conv2_files.split(",") if args.conv2_files else []
    global output_folder, metadata
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # ---- metadata load / estimation ----
    metadata = _try_load_metadata(output_folder)
    if metadata is None:
        props = (
            None
            if args.main_files_proportions is None
            else [float(x) for x in args.main_files_proportions.split(",")]
        )
        metadata = _estimate_metadata(
            output_folder,
            args.samples_per_file,
            args.T_final,
            main_paths,
            conv1_paths,
            conv2_paths,
            args.add_long_dep_1,
            args.add_long_dep_2,
            props,
        )
        save_metadata(output_folder, metadata)

    # Ensure list lengths match current files (handles restored metadata)
    ensure_list(metadata, "main_files_offsets", len(main_paths), 0)
    ensure_list(metadata, "main_files_cycles", len(main_paths), 0)
    ensure_list(metadata, "conv1_indices", len(conv1_paths), 0)
    ensure_list(metadata, "conv1_cycles", len(conv1_paths), 0)
    ensure_list(metadata, "conv2_indices", len(conv2_paths), 0)
    ensure_list(metadata, "conv2_cycles", len(conv2_paths), 0)

    # ---- log metadata at startup ----
    logging.info("Metadata at startup:\n%s", json.dumps(metadata, indent=2, ensure_ascii=False))

    sp = setup_tokenizer(args.tokenizer_model)

    global conv1_fhs, conv2_fhs
    conv1_fhs = [open(p, "r", encoding="utf-8") for p in conv1_paths]
    conv2_fhs = [open(p, "r", encoding="utf-8") for p in conv2_paths] if conv2_paths else []

    # proportions
    if args.main_files_proportions is None:
        props = [1.0] * len(main_paths)
    else:
        raw = args.main_files_proportions.split(",")
        if len(raw) != len(main_paths):
            raise ValueError("Mismatch between main_files and proportions.")
        props = [float(x.strip()) for x in raw]
    props = [p / sum(props) for p in props]

    main_q = queue.Queue(maxsize=args.buffer_size)
    out_q = queue.Queue(maxsize=1000)
    meta_lock = threading.Lock()

    main_fhs = [open(p, "rb") for p in main_paths]
    for i, fh in enumerate(main_fhs):
        skip_main_offset(fh, metadata["main_files_offsets"][i], meta_lock, i)

    lines_read_total = 0
    read_bar = tqdm(
        total=args.max_main_lines,
        desc="Reading main data",
        position=0,
        leave=True,
        unit="lines",
    )

    def proportion_reader():
        nonlocal lines_read_total
        while lines_read_total < args.max_main_lines:
            idx = random.choices(range(len(main_paths)), weights=props, k=1)[0]
            with meta_lock:
                line = read_line_cycling(main_fhs[idx], idx, meta_lock)
                if not line:
                    continue
            if len(line) > MAX_LINE_SIZE:
                continue
            txt = line.decode("utf-8", errors="replace").strip()
            try:
                main_q.put(txt, timeout=QUEUE_PUT_TIMEOUT)
            except queue.Full:
                continue
            read_bar.update(1)
            lines_read_total += 1
            if lines_read_total >= args.max_main_lines:
                break
        for _ in range(args.num_worker_threads):
            main_q.put(None)

    worker_bars = []
    for i in range(args.num_worker_threads):
        bar = tqdm(
            total=args.T_final,
            desc=f"Worker {i} tokens",
            position=1 + i,
            leave=True,
            unit="tokens",
        )
        bar.reset(0)
        worker_bars.append(bar)

    writer_pos = 1 + args.num_worker_threads
    writer_bar = tqdm(
        desc="Writing output", position=writer_pos, leave=True, unit="samples"
    )

    def worker_func(wid):
        bar = worker_bars[wid]
        while True:
            conversation = []
            tok_count = 0
            lines_used = 0
            bar.total = args.T_final
            bar.reset(0)
            while True:
                try:
                    line = main_q.get(timeout=QUEUE_PUT_TIMEOUT)
                except queue.Empty:
                    break
                if line is None:
                    main_q.put(None)
                    return
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                # new_turns = data.get("conversations", [])
                # the incoming data is in oai format, so we need to use the messages field
                new_turns = data.get("messages", [])
                if not new_turns[0].get("role") == "system": # the first turn is a system turn.
                    continue
                new_turns = new_turns[1:] # remove the system turn
                
                needed = count_turn_list_tokens_oai(new_turns, sp)
                pot = tok_count + needed
                if pot <= args.T_main:
                    # conversation = merge_conversations(conversation, new_turns)
                    conversation = merge_conversations_oai(conversation, new_turns)
                    tok_count = pot
                    if needed:
                        bar.update(needed)
                    lines_used += 1
                else:
                    if args.mode == "allow_overflow":
                        # conversation = merge_conversations(conversation, new_turns)
                        conversation = merge_conversations_oai(conversation, new_turns)
                        bar.update(needed)
                        tok_count += needed
                        lines_used += 1
                    break
            if lines_used == 0 and not conversation:
                try:
                    line2 = main_q.get(timeout=1)
                    if line2 is None:
                        main_q.put(None)
                        return
                    main_q.put(line2)
                    continue
                except queue.Empty:
                    return

            system_prompt = ""
            if system_prompt:
                sample = conversation
                dep1c = dep2c = 0
            else:
                sample, newcount, d1s, d2s = insert_dependencies_until_threshold_partial(
                    conversation,
                    tok_count,
                    sp,
                    args.T_final,
                    args.add_long_dep_1,
                    args.add_long_dep_2,
                    get_dep_line_dep1,
                    get_dep_line_dep2,
                    args.num_ref_turns,
                )
                if newcount > tok_count:
                    bar.update(newcount - tok_count)
                tok_count = newcount
                dep1c = d1s
                dep2c = d2s

            # final_conv = {
            #     "system": system_prompt,
            #     "mask": "User",
            #     "conversations": sanitize_conversation(sample),
            # }
            messages = sanitize_conversation_oai(sample)
            first_message = {"role": "system", "content": system_prompt}
            messages = [first_message] + messages
            final_conv = {
                "messages": messages,
            }
            ctk = count_turn_list_tokens_oai(final_conv["messages"], sp)
            with meta_lock:
                update_stats(metadata, ctk, args.T_final)
                k = f"dep1:{dep1c}_dep2:{dep2c}"
                metadata["dependency_stats"][k] = metadata["dependency_stats"].get(
                    k, 0
                ) + 1
                save_metadata(output_folder, metadata)
            try:
                out_q.put(final_conv, timeout=QUEUE_PUT_TIMEOUT)
            except queue.Full:
                pass

    def writer_func():
        out_idx = metadata.get("current_output_file_index", 1)
        outf = open_output_file(output_folder, out_idx)
        in_file = 0
        finished = 0
        while True:
            try:
                item = out_q.get(timeout=QUEUE_PUT_TIMEOUT)
            except queue.Empty:
                continue
            if item is None:
                finished += 1
                if finished == args.num_worker_threads:
                    break
                continue
            outf.write(json.dumps(item, ensure_ascii=False) + "\n")
            outf.flush()
            os.fsync(outf.fileno())
            in_file += 1
            writer_bar.update(1)
            if in_file >= args.samples_per_file:
                outf.close()
                in_file = 0
                with meta_lock:
                    out_idx += 1
                    metadata["current_output_file_index"] = out_idx
                    save_metadata(output_folder, metadata)
                outf = open_output_file(output_folder, out_idx)
        outf.close()

    threading.Thread(target=proportion_reader, daemon=True).start()

    workers = []
    for w in range(args.num_worker_threads):
        t = threading.Thread(target=worker_func, args=(w,), daemon=True)
        workers.append(t)
        t.start()

    writer_t = threading.Thread(target=writer_func, daemon=True)
    writer_t.start()

    for w in workers:
        w.join()

    for _ in range(args.num_worker_threads):
        out_q.put(None)
    writer_t.join()

    read_bar.close()

    for fh in main_fhs:
        fh.close()
    for fh in conv1_fhs:
        fh.close()
    for fh in conv2_fhs:
        fh.close()

    with meta_lock:
        s = metadata["stats"]
        if s["total_conversations"]:
            s["mean_tokens"] = s["total_tokens"] / s["total_conversations"]
        save_metadata(output_folder, metadata)


if __name__ == "__main__":
    main()
