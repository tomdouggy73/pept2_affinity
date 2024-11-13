import time
import subprocess

# Delay for 1.5 hours (90 minutes)
time.sleep(90 * 60)  # 90 minutes * 60 seconds per minute

# Run the second script after the delay
subprocess.run(['python', 'predict_zinc_database.py'])
