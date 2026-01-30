import csv
def parse_csv_results(csv_path):
    """
    Read the CSV results and convert string representations of lists back to actual lists.
    """
    results = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string representations of lists back to actual lists
            row['x0'] = [float(x) for x in row['x0'].split(',')]
            row['optimized_x'] = [float(x) for x in row['optimized_x'].split(',')]

            # Convert other fields to appropriate types
            row['optim_idx'] = int(row['optim_idx'])
            row['nll'] = float(row['nll'])
            row['success'] = row['success'].lower() == 'true'
            row['bic'] = float(row['bic'])

            # Handle empty test_sessions
            if row['test_sessions'] == '':
                row['test_sessions'] = None
            else:
                # If test_sessions contains data, parse it appropriately
                # For now, keeping as string since the original code shows it can be empty or contain session data
                pass

            results.append(row)

    return results

# Example usage:
if __name__ == '__main__':
    # Replace with your actual CSV path
    csv_path = "S:/fileTransferFromWindows/output_gen_2026-01-21_10-29-04.csv"
    parsed_results = parse_csv_results(csv_path)

    # Now you can access the data as proper Python types
    for result in parsed_results:
        print(f"Animal: {result['animal']}")
        print(f"Optimization index: {result['optim_idx']}")
        print(f"Initial parameters: {result['x0']}")
        print(f"Optimized parameters: {result['optimized_x']}")
        print(f"NLL: {result['nll']}")
        print(f"Success: {result['success']}")
        print(f"BIC: {result['bic']}")
        print("---")

    import pandas as pd
    results_df = pd.DataFrame(parsed_results)
    results_df = results_df.sort_values(by = ["animal", "optim_idx"])
    
    import matplotlib.pyplot as plt
    import numpy as np
    plt.bar(x = np.arange(results_df.animal.unique().shape[0]), height = results_df.groupby('animal').nll.min())