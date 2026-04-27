"""
Login to HuggingFace using token from .env file.

Prerequisites:
1. Install: pip install huggingface_hub python-dotenv
2. Make sure HF_TOKEN is set in .env file

Usage:
    python hf_login.py
"""

import os
from huggingface_hub import login
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get HF token from environment
    hf_token = os.getenv('HF_TOKEN')

    if not hf_token:
        print("❌ Error: HF_TOKEN not found in .env file")
        print("\nPlease add your HuggingFace token to .env:")
        print('HF_TOKEN="hf_..."')
        return

    print("Logging in to HuggingFace...")
    try:
        login(token=hf_token, add_to_git_credential=True)
        print("✅ Successfully logged in!")
        print("\nYou can now:")
        print("  • Run python sync_questions_to_hf.py")
        print("  • Upload datasets to HuggingFace")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        print("\nMake sure your HF_TOKEN in .env is valid")

if __name__ == "__main__":
    main()