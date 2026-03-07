import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables from .env file
load_dotenv()

# Get credentials and config from environment
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")

if not HF_TOKEN or not HF_REPO_ID:
    print("Error: Please make sure HF_TOKEN and HF_REPO_ID are set in your .env file.")
    exit(1)

api = HfApi(token=HF_TOKEN)

print(f"Uploading data to Hugging Face repository: {HF_REPO_ID} ...")
print("This may take some time depending on your internet connection (approx. 11GB).")
print("Using upload_large_folder for better stability.")

try:
    # Use upload_large_folder for stability on 11GB. It creates incremental commits.
    api.upload_large_folder(
        folder_path="data",
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )
    print("Upload completed successfully!")
except Exception as e:
    print(f"An error occurred during upload: {e}")
