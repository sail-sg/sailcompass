from huggingface_hub import snapshot_download

snapshot_download(repo_id= "sail/Sailcompass_data", 
    repo_type='dataset',
    local_dir= 'data/',
    local_dir_use_symlinks=False,
)