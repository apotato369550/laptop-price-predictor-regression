import kaggle

kaggle.api.authenticate()

# Download latest version
kaggle.api.dataset_download_files("owm4096/laptop-prices", path='.', unzip=True)
