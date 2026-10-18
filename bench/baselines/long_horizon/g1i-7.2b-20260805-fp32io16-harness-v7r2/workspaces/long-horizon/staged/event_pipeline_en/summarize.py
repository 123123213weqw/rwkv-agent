import csv
import json
import sys
from collections import defaultdict
def main():
    if len(sys.argv) != 4:
        print("Usage: python3 summarize.py input.csv output.json checksum.txt")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    checksum_path = sys.argv[3]
    team_points = defaultdict(int)
    seen_ids = set()
    try:
        with open(input_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    team = row['team']
                    points = int(row['points'])
                except (KeyError, ValueError):
                    continue
                if team not in team_points:
                    team_points[team] = 0
                team_points[team] += points
                seen_ids.add(row['id'])
    except FileNotFoundError:
        print(f"Error: {input_path} not found")
        sys.exit(1)
    valid_events = len(seen_ids)
    teams = {team: points for team, points in team_points.items()}
    sorted_teams = sorted(teams.items(), key=lambda x: x[0])
    output_data = {
        'teams': {team: points for team, points in sorted_teams},
        'valid_events': valid_events
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    with open(checksum_path, 'w') as f:
        f.write(f"{valid_events}\n")
if __name__ == '__main__':
    main()