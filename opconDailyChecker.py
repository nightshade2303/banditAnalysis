import os
import pandas as pd
import subprocess
import schedule
import time


# Define paths
script_path = r"C:\Users\dlab\rishika_sim\sessdfGenerator_daily.py"
output_file = r"L:\4portProb_processed\sessdf_daily.csv"
data_directory = r"L:\4portProb"

def find_most_recent_file(directory):
    """Find the most recent file in the given directory."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def run_script_and_check_duplicates():

    # Run the Python script
    try:
        print("Running the Python script...")
        subprocess.run(["python", script_path], check=True)
        print("Script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running the script: {e}")
        return

    # Check for duplicated trials in the output file
    try:
        print("Checking for duplicated trials...")
        sessdf = pd.read_csv(output_file)
        duplicated_counts = sessdf.groupby('animal').apply(
            lambda group: group.duplicated(subset=['session', 'trialstart', 'trialend', 'eptime']).sum(),
            include_groups = True
        )
        for animal, duplicated_count in duplicated_counts.items():
            print(f"Animal: {animal}, Number of duplicated trials: {duplicated_count}, Latest file: {sessdf[sessdf.animal == animal].datetime.max()}")

    except Exception as e:
        print(f"Error processing the output file: {e}")

# Schedule the task to run daily
schedule.every().day.at("14:52").do(run_script_and_check_duplicates)  # Adjust the time as needed

print("Scheduler is running. Press Ctrl+C to exit.")
while True:
    schedule.run_pending()
    time.sleep(1)
