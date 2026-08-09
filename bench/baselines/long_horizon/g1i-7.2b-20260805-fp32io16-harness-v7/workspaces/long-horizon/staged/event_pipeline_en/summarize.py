import csv
import json
import sys

def main():
    csv_path = sys.argv[1]
    json_path = sys.argv[2]
    checksum_path = sys.argv[3]

    teams = {}
    valid_events = 0

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points = int(row['points'])
            except ValueError:
                continue
            team = row['team']
            if team not in teams:
                teams[team] = points
            else:
                teams[team] = min(teams[team], points)
            valid_events += 1

    sorted_teams = sorted(teams.items(), key=lambda x: x[0])
    payload = {
        'teams': {k: v for k, v in sorted_teams},
        'valid_events': valid_events
    }

    with open(json_path, 'w') as f:
        json.dump(payload, f)

    with open(checksum_path, 'w') as f:
        f.write(f"{hash(json.dumps(payload, sort_keys=True).encode())}\n")

if __name__ == '__main__':
    main()