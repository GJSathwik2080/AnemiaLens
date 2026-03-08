import pandas as pd

for mod in ["conjunctiva", "nail", "palm"]:
    csv_path = f"data/{mod}/labels.csv"
    df = pd.read_csv(csv_path)
    total = len(df)
    anemic = (df["hb_level"] < 11.0).sum()
    healthy = (df["hb_level"] >= 11.0).sum()

    print(f"\n{mod.upper()}")
    print(f"  Total   : {total}")
    print(f"  Anemic  : {anemic}")
    print(f"  Healthy : {healthy}")

    if anemic == 0:
        print(f"  ❌ ZERO healthy images! Model cannot learn!")
    elif healthy == 0:
        print(f"  ❌ ZERO anemic images! Model cannot learn!")
    else:
        print(f"  ✅ Both classes present")