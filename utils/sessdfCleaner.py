from dfLoading import *
import subprocess
import schedule
import time

# Define paths
script_path = r"C:\Users\dlab\rishika_sim\sessdfGenerator_v2_20241211.py"
output_file = r"L:\4portProb_processed\sessdf.csv"
data_directory = r"L:\4portProb"

def run_sessdfgen(load_df = False):
    # Run the Python script
    if load_df==True:
        try:
            print("Running the Python script...")
            subprocess.run(["python", script_path], check=True)
            print("Script executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running the script: {e}")
            return
    
    try:
        df = read_df(output_file)
        df = remove_duplicates(df, trial_limit=5)
        df.to_pickle(data_directory+r"\cleandf.pkl")
    except:
        print("trouble reading the file")
    return

# Schedule the task to run weekly
schedule.every().thursday.do(run_sessdfgen)  # Adjust the day as needed

print("Scheduler is running. Press Ctrl+C to exit.")
while True:
    schedule.run_pending()
    time.sleep(1)
