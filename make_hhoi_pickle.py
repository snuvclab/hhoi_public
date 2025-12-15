import argparse
import pickle
import sys
import re
import os
from collections import defaultdict, deque

def parse_hhi_pairs(pair_strings):
    pairs = []
    pair_pattern = re.compile(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?")
    for pair_str in pair_strings:
        match = pair_pattern.fullmatch(pair_str)
        if not match:
            raise ValueError(f"Invalid pair format: '{pair_str}'. Use (1,2) or 1,2 format.")
        idx1, idx2 = int(match.group(1)), int(match.group(2))
        pairs.append((idx1, idx2))
    return pairs

def check_dag(num_humans, hhi_pairs):
    """Check if the HHI pairs form a DAG (no cycles)."""
    graph = defaultdict(list)
    indegree = [0] * (num_humans + 1)  # 1-based index

    for u, v in hhi_pairs:
        graph[u].append(v)
        indegree[v] += 1

    # Kahn's algorithm for cycle detection
    queue = deque([node for node in range(1, num_humans + 1) if indegree[node] == 0])
    visited_count = 0

    while queue:
        node = queue.popleft()
        visited_count += 1
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if visited_count != num_humans:
        raise ValueError("Error: HHI pairs contain a cycle. The pairs must form a DAG.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create a pickle file for human interactions (HOI, HHI).")
    parser.add_argument('--num_humans', type=int, required=True, help='Number of humans.')
    parser.add_argument('--hoi_texts', nargs='+', required=True, help='List of HOI texts (one per human).')
    parser.add_argument('--hhi_pairs', nargs='+', help="List of HHI pairs like '(1,2)' '(2,3)'.")
    parser.add_argument('--hhi_texts', nargs='+', help='List of HHI texts (one per HHI pair).')
    parser.add_argument('--out_file_name', type=str, required=True, help='Output pickle file name.')
    args = parser.parse_args()

    if len(args.hoi_texts) != args.num_humans:
        print("Error: Number of HOI texts must match the number of humans.", file=sys.stderr)
        sys.exit(1)

    if args.hhi_pairs is not None:
        pairs = parse_hhi_pairs(args.hhi_pairs)
        if args.hhi_texts is None or len(args.hhi_texts) != len(pairs):
            print("Error: Number of HHI texts must match the number of HHI pairs.", file=sys.stderr)
            sys.exit(1)
        check_dag(args.num_humans, pairs)
        args.hhi_pairs = pairs
    else:
        args.hhi_pairs = []
        args.hhi_texts = []

    return args

def main():
    args = parse_arguments()

    human_list = [f"human{i+1}" for i in range(args.num_humans)]
    hoi_text_list = args.hoi_texts

    hhi_pair_list = [(f"human{idx1}", f"human{idx2}") for idx1, idx2 in args.hhi_pairs]
    hhi_text_list = args.hhi_texts

    data = {
        "human_list": human_list,
        "hoi_text_list": hoi_text_list,
        "hhi_text_list": hhi_text_list,
        "hhi_pair_list": hhi_pair_list
    }

    pickle_dir = "data/pickle"
    os.makedirs(pickle_dir, exist_ok=True)
    with open(f"{pickle_dir}/{args.out_file_name}.pkl", "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()