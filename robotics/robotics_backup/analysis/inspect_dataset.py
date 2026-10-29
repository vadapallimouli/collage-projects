import pickle

DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"

with open(DATASET_PATH, "rb") as f:
    data = pickle.load(f)

problem = data["tabletop"]["task_oriented"][0]

print("Problem type:")
print(type(problem))

print("\nProblem attributes:")

if hasattr(problem, "__dict__"):
    for key, value in problem.__dict__.items():
        print(f"\n{key}")
        print("  Type:", type(value))

        if hasattr(value, "shape"):
            print("  Shape:", value.shape)

        elif hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            try:
                print("  Length:", len(value))
            except Exception:
                pass

else:
    print(problem)