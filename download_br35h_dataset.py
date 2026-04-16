import os
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "brain_tumor_dataset")
KAGGLEHUB_CACHE = os.path.join(BASE_DIR, ".kagglehub_cache")
KAGGLE_DATASET = "ahmedhamada0/brain-tumor-detection"


def main():
    os.environ["KAGGLEHUB_CACHE"] = KAGGLEHUB_CACHE

    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Path to dataset files:", path)

    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    os.makedirs(DATASET_DIR, exist_ok=True)
    for class_name in ("no", "yes"):
        src = os.path.join(path, class_name)
        dst = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(src):
            raise RuntimeError(f"Missing expected folder: {src}")
        shutil.copytree(src, dst)

    print("Copied Br35H no/yes folders to:", DATASET_DIR)


if __name__ == "__main__":
    main()
