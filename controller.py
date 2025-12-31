import subprocess
import time
import os

# =========================================================================================
# 1. PIPELINE CONFIGURATION
# =========================================================================================
# Base models to train
BASE_MODELS = [
    "lgbmodel.py",
    "xgboost.py",
    "catboostmodel.py",
    "logisticregression.py",
    "extratree.py",
    "mlpmodel.py",
    "tabnetmodel.py",
    "knnmodel.py"
]

# Post-processing and Ensembling scripts
POST_PROCESSING = [
    "ensemble.py",
    "blender.py"
]

def run_script(script_name):
    """Executes a python script and logs output in real-time."""
    print(f"\n{'='*60}")
    print(f"PROCESS: {script_name}")
    print(f"START TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        # Using 'python' - change to 'python3' if required by your environment
        process = subprocess.Popen(["python", script_name], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.STDOUT, 
                                   text=True)
        
        # Stream output to terminal
        for line in process.stdout:
            print(line, end="")
        
        process.wait()
        
        elapsed = (time.time() - start_time) / 60
        if process.returncode == 0:
            print(f"\n✅ SUCCESS: {script_name} finished in {elapsed:.2f} minutes.")
            return True
        else:
            print(f"\n❌ FAILED: {script_name} exited with code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"\n⚠️ ERROR running {script_name}: {str(e)}")
        return False

# =========================================================================================
# 2. EXECUTION ENGINE
# =========================================================================================
if __name__ == "__main__":
    total_start = time.time()
    
    # Pre-flight check: Create necessary folders
    for folder in ['data', 'oof', 'submissions']:
        os.makedirs(folder, exist_ok=True)
    
    print(f"🚀 Starting Pipeline: {len(BASE_MODELS)} base models followed by ensembling.")

    # Step 1: Run Base Models
    successful_models = 0
    for script in BASE_MODELS:
        if os.path.exists(script):
            if run_script(script):
                successful_models += 1
        else:
            print(f"⏩ Skipping {script}: File not found.")

    # Step 2: Run Analysis and Blender (only if we have OOFs)
    if successful_models > 0:
        print(f"\n{'*'*60}")
        print(f"BASE MODELS COMPLETE ({successful_models}/{len(BASE_MODELS)} success)")
        print(f"STARTING ENSEMBLE PHASE")
        print(f"{'*'*60}\n")
        
        for script in POST_PROCESSING:
            if os.path.exists(script):
                run_script(script)
            else:
                print(f"⏩ Skipping {script}: File not found.")
    else:
        print("\n🛑 CRITICAL FAILURE: No base models completed. Aborting ensemble phase.")

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"PIPELINE FINISHED")
    print(f"TOTAL TIME: {total_elapsed:.2f} minutes")
    print(f"{'='*60}")