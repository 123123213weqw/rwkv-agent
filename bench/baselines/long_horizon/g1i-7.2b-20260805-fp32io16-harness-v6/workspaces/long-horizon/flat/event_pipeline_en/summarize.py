#!/usr/bin/env python3
import csv
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: summarize.py <input> <output> <checksum>")
        sys.exit(1)
    input_path, output_path, checksum_path = sys.argv[1:4]
    teams = {}
    valid_events = 0
    with open(input_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points = int(row['points'])
            except ValueError:
                continue
            team = row['team']
            if team not in teams:
                teams[team] = 0
            teams[team] += points
            valid_events += 1
    sorted_teams = sorted(teams.keys())
    summary = {"teams": {t: teams[t] for t in sorted_teams}, "valid_events": valid_events}
    with open(output_path, 'w') as f:
        json.dump(summary, f)
    with open(checksum_path, 'w') as f:
        f.write(hash(summary) & 0xFFFFFFFFFFFFFFFF)

if __name__ == '__main__':
    main()