import os
import pickle
from multiprocessing import Pool
import shutil
from collections import defaultdict
import regex as re
from tqdm import tqdm
import numpy as np

def pretokenize(text): # 사전토큰화
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for m in re.finditer(pattern, text):
        yield m.group(0)

def count_pairs(ids, weight = 1, counts = None):
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += weight

    return counts

def merge(ids, pair, new_id):
    merged_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids

def find_chunk_boundaries(file_path, num_chunks, end_token = "<|endoftext|>"):
    byte_end_token = end_token.encode("utf-8")

    with open(file_path, "rb") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // num_chunks
        chunk_boundaries = [i * chunk_size for i in range(num_chunks)]
        chunk_boundaries.append(file_size)

        buffer_size = 4096

        for bi in range(1, len(chunk_boundaries) - 1):
            chunk_position = chunk_boundaries[bi] # chunk를 어디서 자를지
            file.seek(chunk_position)

            while True:
                buffer = file.read(buffer_size)

                if buffer == b"": # 파일의 맨끝에 도착하면
                    chunk_boundaries[bi] = file_size
                    break

                end_position = buffer.find(byte_end_token) # 종료 토큰을 찾게함
                if end_position != -1:
                    chunk_boundaries[bi] = chunk_position + end_position
                    break

                chunk_position += buffer_size # if에서 걸리지않으면 다음 포지션으로 이동

    return sorted(set(chunk_boundaries)) # chunk bondary의 중복값을 제거해서 확인

def process_single_chunk(file_path, start, end, end_token): # 하나의 chunk를 처리하는 함수
    pretoken_counts = defaultdict(int)

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode("utf-8", errors = "ignore")
        texts = chunk_text.split(end_token)

        for text in texts:
            for pretoken in pretokenize(text):
                pretoken_counts[pretoken] += 1

    return pretoken_counts

def pretoken_chunk(args):
    file_path, start, end, end_token = args
    pretoken_counts = defaultdict(int)

    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode('utf-8', errors = 'ignore')
        texts = chunk_text.split(end_token)

        for text in texts:
            for pretoken in pretokenize(text):
                pretoken_counts[pretoken] += 1

    return pretoken_counts

def train_bpe(file_path,
              vocab_size,
              end_token = "<|endoftext|>",
              num_processes = 8,
              num_chunks = 8):
    chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
    total_chunks = len(chunk_boundaries) - 1
    chunk_info_list = []

    for i in range(total_chunks):
        start = chunk_boundaries[i]
        end = chunk_boundaries[i + 1]
        chunk_info_list.append((file_path, start, end, end_token))

    with Pool(processes = num_processes) as pool:
        all_results = list(tqdm(pool.imap(pretoken_chunk, chunk_info_list),
                                total = len(chunk_info_list),
                                desc = "Pretokenizing"))
    pretoken_counts = defaultdict(int)
    for chunk_result in all_results:
        for pretoken, count in chunk_result.items():
            pretoken_counts[pretoken] += count

    ids_counts = {tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}

    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    pair_to_ids = defaultdict(set)

    pair_counts = defaultdict(int)
    for ids, count in ids_counts.items():
        count_pairs(ids, count, pair_counts)
        for pair in zip(ids, ids[1:]):
            pair_to_ids[pair].add(ids)

    for step in tqdm(range(num_merges), desc = "Training BPE"):
        if not pair_counts:
            break

        best_pair = max(pair_counts,
                        key = lambda pair: (pair_counts[pair],
                                            pair[0], pair[1]))
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        affected_ids = pair_to_ids[best_pair]
        del pair_to_ids[best_pair] #사용하지 않는 것 삭제

        for ids in affected_ids:
            ids_count = ids_counts[tuple(ids)]
            new_ids = merge(ids, best_pair, new_id)

            del ids_counts[tuple(ids)]
            ids_counts[tuple(new_ids)] = ids_count

            old_counts = count_pairs(ids)
            for pair, count in old_counts.items():
                pair_counts[pair] -= count * ids_count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_ids[pair].discard(tuple(ids))

            new_counts = count_pairs(new_ids)
            for pair, count in new_counts.items():
                pair_counts[pair] += count * ids_count
                pair_to_ids[pair].add(tuple(new_ids))
    return merge_rules

class BPETokenizer:
    def __init__(self, merge_rules, end_token = "<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)

        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]

        self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")

        self.vocab_size = len(self.id_to_bytes)

    @staticmethod
    def load_from(file_path):
        with open(file_path, "rb") as f:
            merge_rules = pickle.load(f)
        return BPETokenizer(merge_rules)

    def _encode_text(self, text):
        ids = list(text.encode("utf-8"))

        def get_merge_priority(pair): # 우선순위 추가
            return self.merge_rules.get(pair, float("inf"))

        while len(ids) > 1:
            counts = count_pairs(ids)
            best_pair = min(counts, key = get_merge_priority)

            if best_pair not in self.merge_rules:
                break

            new_id = self.merge_rules[best_pair]
            ids = merge(ids, best_pair, new_id)

        return ids

    def encode(self, input_text, show_progress = False): # encode - text
        pattern = "(" + re.escape(self.end_token) + ")"
        texts = re.split(pattern, input_text)
        all_ids = []
        texts = tqdm(texts, desc = "Encoding") if show_progress else texts
        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                for pretoken in pretokenize(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)
        return all_ids

    def _encode_chunk(self, args):
        file_path, start, end, cache_dir, chunk_ids = args

        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_byte = f.read(end - start)
            chunk_text = chunk_byte.decode("utf-8", errors = "ignore")
            ids = self.encode(chunk_text)

        cache_file = os.path.join(cache_dir, f"chunk_{chunk_ids:05d}.npy")
        np.array(ids, dtype = np.uint16).tofile(cache_file)

        return cache_file, len(ids)

    def encode_file(self,
                    file_path,
                    output_file,
                    num_processes = 4,
                    num_chunks = 64,
                    cache_dir = "bpe_cache"):
        os.makedirs(cache_dir, exist_ok = True)

        try:
            chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
            total_chunks = len(chunk_boundaries) - 1
            chunk_info_list = []
            for i in range(total_chunks):
                start = chunk_boundaries[i]
                end = chunk_boundaries[i + 1]
                chunk_info_list.append((file_path, start, end, cache_dir, i))

            with Pool(processes = num_processes) as pool:
                cache_results = list(tqdm(pool.imap(self._encode_chunk, chunk_info_list),
                                          total = len(chunk_info_list),
                                          desc = "Encoding chunks"))

            cache_files = [r[0] for r in cache_results]
            token_counts = [r[1] for r in cache_results]
            total_tokens = sum(token_counts)

            dtype = np.uint16
            arr = np.memmap(output_file,
                            dtype = dtype,
                            mode = "w+",
                            shape = (total_tokens, ))
            idx = 0
            for cache_file in cache_files:
                chunk_data = np.fromfile(cache_file, dtype = dtype)
                arr[idx: idx + len(chunk_data)] = chunk_data
                idx += len(chunk_data)

            arr.flush()
            del arr
        finally:
            shutil.rmtree(cache_dir)
        return total_tokens

    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors = "replace")
        return text