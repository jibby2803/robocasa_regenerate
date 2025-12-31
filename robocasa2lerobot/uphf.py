from huggingface_hub import HfApi
from dotenv import load_dotenv
load_dotenv()

repo_id = "binhng/robocasa_5_100demos_chosen_tasks_regenerated_element_ooi_v1"
local_folder = "/home/binhng/Workspace/robocasa/robocasa/datasets/regenerate/robocasa-100demos-5chosen-tasks"

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Initial commit",
    # optional filters
    ignore_patterns=["**/__pycache__/**", "**/*.tmp", "**/.ipynb_checkpoints/**"],
    # allow_patterns=["**/*.pt","**/*.json"]  # alternatively, whitelist
)
 