from huggingface_hub import HfApi

# upload fractial_mixes.zip and results.zip to HF Hub in manu/IFTEval

api = HfApi()
api.upload_file(path_in_repo="fractial_mixes.zip", repo_id="manu/IFTEval", path_or_fileobj="fractial_mixes.zip", repo_type="dataset")
api.upload_file(path_in_repo="results.zip", repo_id="manu/IFTEval", path_or_fileobj="results.zip", repo_type="dataset")
api.upload_file(path_in_repo="README.md", repo_id="manu/IFTEval", path_or_fileobj="README.md", repo_type="dataset")

print("Upload Successful!")
