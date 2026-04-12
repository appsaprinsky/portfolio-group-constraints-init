# functions/group_constraints.py

import pandas as pd


class GroupConstraints:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.groups, self.limits = self._load_from_csv()

    def _load_from_csv(self):
        df = pd.read_csv(self.csv_path)

        required_cols = {"group", "tickers", "min", "max"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        groups = {}
        limits = {}

        for _, row in df.iterrows():
            group_name = row["group"]

            tickers = [t.strip() for t in row["tickers"].split("|")]

            min_val = float(row["min"])
            max_val = float(row["max"])

            # sanity checks (VERY important)
            if min_val < 0 or max_val > 1:
                raise ValueError(f"{group_name}: bounds must be in [0,1]")

            if min_val > max_val:
                raise ValueError(f"{group_name}: min > max")

            groups[group_name] = tickers
            limits[group_name] = {
                "min": min_val,
                "max": max_val
            }

        return groups, limits

    def get_valid_groups(self, valid_tickers):
        filtered_groups = {}
        filtered_limits = {}

        for g, assets in self.groups.items():
            valid_assets = [a for a in assets if a in valid_tickers]

            # only keep groups that still have assets
            if valid_assets:
                filtered_groups[g] = valid_assets
                filtered_limits[g] = self.limits[g]

        return filtered_groups, filtered_limits