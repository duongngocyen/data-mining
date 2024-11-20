import pandas as pd
from collections import defaultdict
from itertools import combinations
import re
import logging
import time
import numpy as np
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_triplegs(df, name, sample_number, timestamp_col='started_at'):
    start_time = time.time()
    logging.info(f"[{name} - Sample {sample_number}] Starting preprocessing of triplegs data")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    start_date = pd.to_datetime('1900-01-01').tz_localize('UTC')
    end_date = start_date + pd.Timedelta(days=30)

    df = df.loc[(df[timestamp_col] >= start_date) & (df[timestamp_col] < end_date)].copy()
    logging.info(f"[{name} - Sample {sample_number}] Filtered data to the first 30 days")

    def linestring_to_int_coords(linestring):
        coords = re.findall(r'(\d+\.\d+ \d+\.\d+)', linestring)
        int_coords = [tuple(map(int, map(float, coord.split()))) for coord in coords]
        return int_coords

    df['geom'] = df['geom'].apply(linestring_to_int_coords)
    logging.info(f"[{name} - Sample {sample_number}] Converted LINESTRING to integer coordinates")

    user_sequences = df.groupby('user_id')['geom'].apply(list).tolist()
    final_output = [coord for sequence in user_sequences for coord in sequence]
    
    end_time = time.time()
    logging.info(f"[{name} - Sample {sample_number}] Completed preprocessing of triplegs in {end_time - start_time:.2f} seconds")
    return final_output

def split_long_triplegs(df, name, sample_number, max_length=1000):
    start_time = time.time()
    logging.info(f"[{name} - Sample {sample_number}] Splitting long triplegs into shorter segments")

    def split_geom(geom, max_length):
        points = geom.split(',')
        return [','.join(points[i:i + max_length]) for i in range(0, len(points), max_length)]

    new_rows = []
    for _, row in df.iterrows():
        geoms = split_geom(row['geom'], max_length)
        for geom in geoms:
            new_row = row.copy()
            new_row['geom'] = geom
            new_rows.append(new_row)

    end_time = time.time()
    logging.info(f"[{name} - Sample {sample_number}] Completed splitting long triplegs in {end_time - start_time:.2f} seconds")
    return pd.DataFrame(new_rows)

def generate_candidates(sequences, length):
    start_time = time.time()
    candidates = set()
    for seq in sequences:
        for i in range(len(seq) - length + 1):
            candidates.add(tuple(seq[i:i + length]))

    end_time = time.time()
    logging.info(f"generate_candidates for length {length}: {(end_time - start_time):.2f} seconds")
    return candidates

def is_subsequence(candidate, sequence):
    it = iter(sequence)
    return all(item in it for item in candidate)

# Triangular matrix for efficient pair counting
def triangular_index(i, j, n):
    """Calculate index in a triangular matrix."""
    if isinstance(i, tuple) or isinstance(j, tuple):
        logging.error("Triangular index function received tuple instead of integers")
        i, j = int(i[0]), int(j[0])  # Attempt to convert if needed
    if i < j:
        i, j = j, i
    return i * (i - 1) // 2 + j

def count_support(sequences, candidates, min_support, n_items):
    """Count support using hashing, bitmap, and triangular matrix."""
    start_time = time.time()

    # Step 1: Hash buckets and bitmap
    bucket_count = 3342277  # Prime number for hashing
    bucket_counts = np.zeros(bucket_count, dtype=int)

    # Populate hash buckets with item pairs
    for seq in sequences:
        for i, j in combinations(seq, 2):
            # Ensure `i` and `j` are integers before hashing
            if isinstance(i, tuple):
                i = int(i[0])
            if isinstance(j, tuple):
                j = int(j[0])
            bucket_counts[hash((i, j)) % bucket_count] += 1

    # Step 2: Bitmap for frequent buckets
    bitmap = bucket_counts >= min_support
    frequent_pairs = {}

    # Triangular matrix for counting pairs
    pair_counts = np.zeros(n_items * (n_items - 1) // 2, dtype=int)

    # Step 3: Count pairs that map to frequent buckets
    for seq in sequences:
        for i, j in combinations(seq, 2):
            if isinstance(i, tuple):
                i = int(i[0])
            if isinstance(j, tuple):
                j = int(j[0])
            if bitmap[hash((i, j)) % bucket_count]:
                idx = triangular_index(i, j, n_items)
                pair_counts[idx] += 1

    # Filter pairs by min_support
    for candidate in candidates:
        if len(candidate) == 2:
            i, j = candidate
            if isinstance(i, tuple):
                i = int(i[0])
            if isinstance(j, tuple):
                j = int(j[0])
            idx = triangular_index(i, j, n_items)
            if pair_counts[idx] >= min_support:
                frequent_pairs[candidate] = pair_counts[idx]

    end_time = time.time()
    logging.info(f"Optimized count_support: {(end_time - start_time):.2f} seconds")
    return frequent_pairs

def prune_candidates(support_count, min_support):
    start_time = time.time()
    pruned_candidates = {seq: count for seq, count in support_count.items() if count >= min_support}
    end_time = time.time()
    logging.info(f"prune_candidates: {(end_time - start_time):.2f} seconds")
    return pruned_candidates

def generate_new_candidates(frequent_sequences, length):
    start_time = time.time()
    new_candidates = set()
    frequent_sequences = list(frequent_sequences)
    for seq1, seq2 in combinations(frequent_sequences, 2):
        if seq1[1:] == seq2[:-1]:
            new_candidates.add(seq1 + (seq2[-1],))
    end_time = time.time()
    logging.info(f"generate_new_candidates for length {length}: {(end_time - start_time):.2f} seconds")
    return new_candidates

def gsp(sequences, name, sample_number, min_support, n_items):
    start_time = time.time()
    logging.info(f"[{name} - Sample {sample_number}] Starting GSP algorithm with min_support={min_support}")
    
    length = 2
    frequent_sequences = generate_candidates(sequences, length)
    all_frequent_sequences = []

    while frequent_sequences:
        # Support counting with optimizations
        support_count = count_support(sequences, frequent_sequences, min_support, n_items)
        
        # Pruning
        frequent_sequences = prune_candidates(support_count, min_support)
        all_frequent_sequences.extend(frequent_sequences.keys())

        # Generating new candidates
        length += 1
        frequent_sequences = generate_new_candidates(frequent_sequences.keys(), length)

    end_time = time.time()
    logging.info(f"[{name} - Sample {sample_number}] Completed GSP algorithm in {end_time - start_time:.2f} seconds")
    return all_frequent_sequences

def save_gsp_results(gsp_results, name, sample_number):
    start_time = time.time()
    logging.info(f"[{name} - Sample {sample_number}] Saving GSP results")

    formatted_sequences = [
        f"• <" + ", ".join(f"{{{', '.join(map(str, elem))}}}" for elem in sequence) + ">"
        for sequence in gsp_results
    ]
    
    output_file = f'gsp_{name}_sample{sample_number}_optimized.csv'
    sequences_df = pd.DataFrame({'Sequence': formatted_sequences})
    sequences_df.to_csv(f"./output/{output_file}", index=False)

    end_time = time.time()
    logging.info(f"[{name} - Sample {sample_number}] GSP results saved to {output_file} in {end_time - start_time:.2f} seconds")

# Define variables for name and sample_number
for name in ['A']:
    sample_number = 10000

    logging.info(f'Dataset {name}, sample={sample_number}')

    # Load the data
    df = pd.read_csv(f'./output/triplegs_{name}.csv').sample(n=10000)

    df = split_long_triplegs(df, name, sample_number)
    sequences = preprocess_triplegs(df, name, sample_number)

    min_support = 2
    n_items = 1000
    logging.info(f"[{name} - Sample {sample_number}] Minimum support threshold: {min_support}")

    frequent_sequences = gsp(sequences, name, sample_number, min_support, n_items)
    logging.info(f"[{name} - Sample {sample_number}] Number of frequent sequences: {len(frequent_sequences)}")

    save_gsp_results(frequent_sequences, name, sample_number)

    logging.info("=============================================")