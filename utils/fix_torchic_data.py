"""
Fix rows in a numpy array where groups of three "81" rows are missing a
fourth row (the one whose 2nd column value from {1, 2, 3, 4} was left out).

Works directly on a numpy array of numbers (ints or floats) rather than
CSV text -- e.g. the array you get from loading the CSV with
np.loadtxt / np.genfromtxt / pandas.to_numpy().
"""

import sys
import numpy as np


TARGET_FIRST_COL = 81
EXPECTED_SECOND_COL_VALUES = {1, 2, 3, 4}


def fix_torchic_rows(data):
    """
    data: a 2D numpy array, at least 2 columns wide, where column 0 is the
    "type" column and column 1 is the "sub-id" column.

    Returns a new 2D numpy array with the missing "81" rows inserted.
    Does not modify `data` in place.
    """
    n = data.shape[0]
    fixed_rows = []  # list of 1D numpy arrays / rows, rebuilt at the end

    i = 0
    while i < n:
        row = data[i]

        if row[0] == TARGET_FIRST_COL:
            # Collect the contiguous run of rows starting with 81
            run_start = i
            j = i
            while j < n and data[j][0] == TARGET_FIRST_COL:
                j += 1
            run = data[run_start:j]

            if len(run) == 3:
                seen_vals = set(int(v) for v in run[:, 1])
                missing = EXPECTED_SECOND_COL_VALUES - seen_vals

                if len(missing) == 1:
                    missing_val = missing.pop()

                    template = run[0].copy()
                    for r in run[1:]:
                        if not np.array_equal(r[2:], template[2:]):
                            sys.stderr.write(
                                "Warning: rows in an 81-run near index "
                                f"{run_start} don't share identical "
                                "trailing columns; using the first row's "
                                "values for the inserted row.\n"
                            )
                            break

                    new_row = template.copy()
                    new_row[0] = TARGET_FIRST_COL
                    new_row[1] = missing_val

                    # Drop the original three rows entirely; keep only
                    # the synthesized missing-value row in their place.
                    fixed_rows.append(new_row)
                else:
                    if missing:
                        sys.stderr.write(
                            f"Warning: 81-run near index {run_start} has "
                            f"unexpected values {seen_vals}; left "
                            "unmodified.\n"
                        )
                    fixed_rows.extend(run)
            else:
                fixed_rows.extend(run)

            i = j
        else:
            fixed_rows.append(row)
            i += 1

    return np.array(fixed_rows)