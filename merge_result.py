import os
import json
import argparse
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge JSON files into a CSV.")
    parser.add_argument('-i', '--input', required=True, help='Input directory containing JSON files.')
    parser.add_argument('-n', '--index', required=True, nargs=2, type=int, help='Index range (inclusive start, exclusive end).')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path.')
    return parser.parse_args()

def check_missing_files(input_dir, start_idx, end_idx):
    missing_files = []
    for idx in range(start_idx, end_idx):
        file_path = os.path.join(input_dir, f"idx_{idx}.json")
        if not os.path.exists(file_path):
            missing_files.append(idx)
    return missing_files

def read_json_files(input_dir, start_idx, end_idx):
    data = []
    for idx in range(start_idx, end_idx):
        file_path = os.path.join(input_dir, f"idx_{idx}.json")
        with open(file_path, 'r', encoding='utf-8') as file:
            data.append(json.load(file))
    return data

def write_to_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, escapechar="\\", quoting=1)

def main():
    args = parse_arguments()

    input_dir = args.input
    start_idx, end_idx = args.index
    output_file = args.output

    if os.path.exists(output_file):
        print(f"Error: Output file '{output_file}' already exists. Exiting without overwriting.")
        return

    missing_files = check_missing_files(input_dir, start_idx, end_idx)
    if missing_files:
        print(f"Error: Missing JSON files for indices: {missing_files}. Exiting.")
        return

    data = read_json_files(input_dir, start_idx, end_idx)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Calculate jailbroken statistics
    total = len(df)
    jailbroken_count = df['jailbroken'].sum()
    jailbroken_ratio = jailbroken_count / total

    print(f"Jailbroken Count: {jailbroken_count}")
    print(f"Jailbroken Ratio: {jailbroken_ratio:.2%}")

    # Write to CSV
    write_to_csv(data, output_file)
    print(f"CSV file '{output_file}' created successfully.")

if __name__ == "__main__":
    main()
