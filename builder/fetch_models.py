from faster_whisper.utils import download_model

model_names = [
    "large-v3",
]


def download_model_weights(selected_model):
    """
    Download model weights.
    """
    print(f"Downloading {selected_model}...")
    download_model(selected_model, cache_dir=None)
    print(f"Finished downloading {selected_model}.")


# Loop through models sequentially
for model_name in model_names:
    download_model_weights(model_name)

print("Finished downloading all models.")
