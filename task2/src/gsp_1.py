import pandas as pd
from collections import defaultdict
from itertools import combinations
import re
import logging
import time

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

def count_support(candidates, sequences):
    start_time = time.time()
    support_count = defaultdict(int)
    for candidate in candidates:
        for seq in sequences:
            if is_subsequence(candidate, seq):
                support_count[candidate] += 1
    end_time = time.time()
    logging.info(f"count_support for {len(candidates)} candidates: {(end_time - start_time):.2f} seconds")
    return support_count

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

def gsp(sequences, name, sample_number, min_support):
    start_time = time.time()
    logging.info(f"[{name} - Sample {sample_number}] Starting GSP algorithm with min_support={min_support}")
    
    length = 2
    frequent_sequences = generate_candidates(sequences, length)
    all_frequent_sequences = []

    while frequent_sequences:
        step_start = time.time()
        
        # Support counting
        support_count = count_support(frequent_sequences, sequences)
        step_end = time.time()
        logging.info(f"[{name} - Sample {sample_number}] Support counting for length {length}: {(step_end - step_start):.2f} seconds")
        
        # Pruning
        step_start = time.time()
        frequent_sequences = prune_candidates(support_count, min_support)
        all_frequent_sequences.extend(frequent_sequences.keys())
        step_end = time.time()
        logging.info(f"[{name} - Sample {sample_number}] Pruning for length {length}: {(step_end - step_start):.2f} seconds")

        # Generating new candidates
        length += 1
        step_start = time.time()
        frequent_sequences = generate_new_candidates(frequent_sequences.keys(), length)
        step_end = time.time()
        logging.info(f"[{name} - Sample {sample_number}] Generating new candidates for length {length}: {(step_end - step_start):.2f} seconds")

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
    
    output_file = f'gsp_{name}_sample{sample_number}.csv'
    sequences_df = pd.DataFrame({'Sequence': formatted_sequences})
    sequences_df.to_csv(f"./output/{output_file}", index=False)

    end_time = time.time()
    logging.info(f"[{name} - Sample {sample_number}] GSP results saved to {output_file} in {end_time - start_time:.2f} seconds")

# Define variables for name and sample_number
name = 'A'
sample_number = 10000

logging.info(f'Dataset {name}, sample={sample_number}')

# Load the data
df = pd.read_csv(f'./output/triplegs_{name}.csv', nrows=sample_number)

df = split_long_triplegs(df, name, sample_number)
sequences = preprocess_triplegs(df, name, sample_number)

min_support = 2
logging.info(f"[{name} - Sample {sample_number}] Minimum support threshold: {min_support}")

frequent_sequences = gsp(sequences, name, sample_number, min_support)
logging.info(f"[{name} - Sample {sample_number}] Number of frequent sequences: {len(frequent_sequences)}")

save_gsp_results(frequent_sequences, name, sample_number)

logging.info("=============================================")
