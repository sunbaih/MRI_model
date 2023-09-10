import os
import shutil

def move_niftis_to_folder():
    # Moves all of the niftis in subfolders in a folder to a different folder
    
    source_folder = "/Users/baihesun/cancer_data/TCGA-GBM_all_data"
    
    destination_folder = "/Users/baihesun/cancer_data/TCGA-GBM_all_niftis"
    
    target_extension = 'ManuallyCorrected.nii.gz'  # Change this to the desired file extension
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(target_extension):
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_folder, file)
                
                
                shutil.copy2(source_file_path, destination_file_path)
                print(f"File '{file}' copied to '{destination_folder}'")

