import os
import pickle


RECORDS_PATH = "data/failure_prediction/trajectory_records.pkl"


def main():
    print("=" * 70)
    print("TRAJECTORY RECORD KEYS")
    print("=" * 70)

    if not os.path.exists(RECORDS_PATH):
        print(f"ERROR: File not found: {RECORDS_PATH}")
        return

    with open(RECORDS_PATH, "rb") as f:
        records = pickle.load(f)

    print(f"\nNumber of records: {len(records)}")

    if not records:
        print("ERROR: No records found.")
        return

    record = records[0]

    print(f"\nNumber of keys in first record: {len(record)}")

    print("\nAll keys:")
    print("-" * 70)

    for i, key in enumerate(record.keys(), start=1):
        value = record[key]

        print(
            f"{i:2d}. {key:<45} "
            f"type={type(value).__name__:<12} "
            f"value={repr(value)[:100]}"
        )

    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)

    successes = sum(r["collision"] == 0 for r in records)
    failures = sum(r["collision"] == 1 for r in records)

    print(f"Successful trajectories : {successes}")
    print(f"Failed trajectories     : {failures}")
    print(f"Total                   : {len(records)}")

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()