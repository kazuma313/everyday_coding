from supervision.assets import download_assets, VideoAssets

def main():
    download_assets(VideoAssets.PEOPLE_WALKING)
    
if __name__ == "__main__":
    main()