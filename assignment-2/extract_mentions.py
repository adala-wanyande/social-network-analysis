import csv
import re
import os
import requests
from collections import defaultdict

def download_file(url, local_filename):
    """
    Downloads a file from a URL if it doesn't already exist locally.
    """
    if not os.path.exists(local_filename):
        print(f"Downloading {local_filename} from {url}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Successfully downloaded {local_filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            return False
    else:
        print(f"{local_filename} already exists.")
    return True

def extract_mention_graph(input_tsv_path, output_csv_path):
    """
    Parses a Twitter TSV file to extract a weighted, directed mention graph
    and saves it as a CSV edge list.

    Args:
        input_tsv_path (str): The path to the input TSV file.
        output_csv_path (str): The path where the output CSV edge list will be saved.
    """
    print(f"\nProcessing {input_tsv_path} to generate mention graph...")

    # The graph is represented as a nested dictionary: {sender: {mentioned_user: count}}
    # Using defaultdict simplifies adding new users and incrementing counts.
    # defaultdict(int) will create a dictionary where missing keys default to an integer value of 0.
    mention_graph = defaultdict(lambda: defaultdict(int))
    
    # This regex is designed to capture valid Twitter usernames mentioned in a tweet.
    # @: Matches the literal '@' symbol.
    # ([A-Za-z0-9_]{1,15}): Captures a group of 1 to 15 characters, which can be
    #                       alphanumeric or an underscore. This matches Twitter's username rules.
    # \b: A word boundary. This is crucial to prevent matching mentions inside email
    #     addresses (e.g., 'user@example.com').
    username_regex = re.compile(r'@([A-Za-z0-9_]{1,15})\b')
    
    lines_processed = 0
    malformed_lines = 0

    try:
        with open(input_tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                lines_processed += 1
                
                # Split the line by the tab character.
                parts = line.strip().split('\t')
                
                # --- Issue Handling: Malformed Lines ---
                # A robust parser must handle lines that don't fit the expected format.
                if len(parts) != 3:
                    malformed_lines += 1
                    continue # Skip this line and move to the next one.
                
                # Unpack the parts into meaningful variables.
                timestamp, sender, content = parts
                
                # --- Issue Handling: Case Sensitivity ---
                # Twitter usernames are case-insensitive. To avoid treating 'UserA' and
                # 'usera' as different nodes, we convert all usernames to a consistent
                # case (e.g., lowercase).
                sender = sender.lower()
                
                # Use the pre-compiled regex to find all mentions in the tweet content.
                # findall() returns a list of all captured groups.
                mentions = username_regex.findall(content)
                
                if not mentions:
                    continue # Skip tweets with no mentions.
                    
                for mentioned_user in mentions:
                    # Also convert the mentioned user to lowercase for consistency.
                    mentioned_user_lower = mentioned_user.lower()
                    
                    # Increment the weight of the directed edge from sender -> mentioned_user.
                    # Thanks to defaultdict, we don't need to check if the keys exist first.
                    mention_graph[sender][mentioned_user_lower] += 1
                    
    except FileNotFoundError:
        print(f"Error: The file {input_tsv_path} was not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    print(f"Finished processing. Total lines: {lines_processed}, Malformed lines skipped: {malformed_lines}")
    
    # --- Step 3: Output the adjacency list as a weighted edge list CSV ---
    print(f"Writing graph to {output_csv_path}...")
    edge_count = 0
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write the header row for clarity.
        writer.writerow(['Source', 'Target', 'Weight'])
        
        # Iterate through the graph structure to write each edge.
        for sender, targets in mention_graph.items():
            for target, weight in targets.items():
                writer.writerow([sender, target, weight])
                edge_count += 1
    
    print(f"Successfully created edge list with {edge_count} unique edges.")


if __name__ == "__main__":
    # --- Define URLs and local filenames ---
    files_to_process = {
        'twitter-small.tsv': 'https://liacs.leidenuniv.nl/~takesfw/SNACS/twitter-small.tsv',
        'twitter-larger.tsv': 'https://liacs.leidenuniv.nl/~takesfw/SNACS/twitter-larger.tsv'
    }

    # --- Download and process each file ---
    for filename, url in files_to_process.items():
        if download_file(url, filename):
            output_filename = filename.replace('.tsv', '-edgelist.csv')
            extract_mention_graph(filename, output_filename)