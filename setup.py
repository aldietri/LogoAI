import subprocess
import shutil
import os 

# Repositories that need to be cloned
repositories = ["https://github.com/matterport/Mask_RCNN", 
                "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix"]

# Files that need to be moved to their proper destinatiuon
filesToDestination = {"testA": "pytorch-CycleGAN-and-pix2pix/datasets/Objects2Icons",
                        "Objects2Icons": "pytorch-CycleGAN-and-pix2pix/checkpoints",
                        "mask_rcnn_coco.h5": "Mask_RCNN"}


print("[System] Beginning to clone repositories.")
for repository in repositories:
    print(f"[System] Currently cloning: {repository}")
    # Define command
    command = f"git clone {repository}"

    # Run command 
    p = subprocess.run(command, shell=True, capture_output=True)

    # Print error code if existent
    if p.returncode != 0:
        print( 'Command:', p.args)
        print( 'exit status:', p.returncode )
        print( 'stdout:', p.stdout.decode() )
        print( 'stderr:', p.stderr.decode() )

print("[System] Finished cloning repositories.")


print("[System] Beginning to move files.")
for src, dest in filesToDestination.items():
    # Create new directory if it doesn't already exist
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    # Move file
    shutil.move(src, dest)

print("[System] Finished moving files.")