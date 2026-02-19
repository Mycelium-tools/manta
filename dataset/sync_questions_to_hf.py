"""
Sync questions from Google Sheets → Local CSV → HuggingFace

Setup (one-time):
1. In Google Sheets: File → Share → Publish to web
2. Select your sheet, choose "Comma-separated values (.csv)"
3. Copy the URL and paste it below as GOOGLE_SHEETS_URL
4. Make sure HF_TOKEN is set in .env file

Usage:
    python sync_questions_to_hf.py

This will:
1. Download latest questions from Google Sheets
2. Save to manta_questions.csv
3. Upload to HuggingFace
"""

import os
import requests
from datasets import load_dataset
from huggingface_hub import create_repo, upload_file, login
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_SHEETS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR_fCnBetwXyIvAVaBJ1Surc6rdzoZm-eR9fdGWni9KvrwetKTpJRfSkyVF7PUfxBNGeI-dGEZp3P7u/pub?output=csv"
LOCAL_CSV = "manta_questions.csv"
HF_DATASET = "mycelium-ai/manta-questions"

def download_from_google_sheets():
    """Download CSV from published Google Sheets."""

    # if GOOGLE_SHEETS_URL:
    #     print("❌ Error: Please set your Google Sheets URL in sync_questions.py")
    #     print("\nSteps:")
    #     print("1. Open your Google Sheet")
    #     print("2. File → Share → Publish to web")
    #     print("3. Select 'Comma-separated values (.csv)'")
    #     print("4. Copy the URL")
    #     print("5. Paste it as GOOGLE_SHEETS_URL in this file")
    #     return False

    print(f"📥 Downloading from Google Sheets...")
    try:
        response = requests.get(GOOGLE_SHEETS_URL)
        response.raise_for_status()

        # Save to local CSV
        with open(LOCAL_CSV, 'wb') as f:
            f.write(response.content)

        # Count lines (questions)
        with open(LOCAL_CSV, 'r') as f:
            num_questions = len(f.readlines()) - 1  # -1 for header

        print(f"✅ Downloaded {num_questions} questions to {LOCAL_CSV}")
        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def upload_to_huggingface():
    """Upload CSV to HuggingFace dataset."""

    if not os.path.exists(LOCAL_CSV):
        print(f"❌ Error: {LOCAL_CSV} not found")
        return False

    # Login to HuggingFace
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print(f"❌ Error: HF_TOKEN not found in .env file")
        return False

    login(token=hf_token)

    # Create repository if it doesn't exist
    print(f"📤 Uploading to HuggingFace ({HF_DATASET})...")
    try:
        create_repo(
            repo_id=HF_DATASET,
            repo_type="dataset",
            exist_ok=True,
            private=False
        )

        # Upload the CSV
        upload_file(
            path_or_fileobj=LOCAL_CSV,
            path_in_repo=LOCAL_CSV,
            repo_id=HF_DATASET,
            repo_type="dataset",
            commit_message="Sync from Google Sheets"
        )

        print(f"✅ Successfully uploaded to https://huggingface.co/datasets/{HF_DATASET}")

        # Verify
        print(f"🔍 Verifying...")
        dataset = load_dataset(HF_DATASET, data_files=LOCAL_CSV, split="train")
        print(f"✅ Verified: {len(dataset)} questions loaded")
        return True

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(f"\n💡 Make sure HF_TOKEN is set in .env file")
        return False

def main():
    print("=" * 60)
    print("Google Sheets → Local CSV → HuggingFace Sync")
    print("=" * 60)

    # Step 1: Download from Google Sheets
    if not download_from_google_sheets():
        return

    # Step 2: Upload to HuggingFace
    if not upload_to_huggingface():
        return

    print("\n" + "=" * 60)
    print("✅ Sync complete!")
    print("=" * 60)
    print(f"\n📊 Your questions are now:")
    print(f"   • Local: {LOCAL_CSV}")
    print(f"   • HuggingFace: https://huggingface.co/datasets/{HF_DATASET}")
    print(f"\nNext time you update your Google Sheet, just run:")
    print(f"   python sync_questions_to_hf.py")

if __name__ == "__main__":
    main()
