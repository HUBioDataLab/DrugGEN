#!/bin/bash
# DrugGEN Setup Script
# Downloads all necessary resources from Google Drive repository

# Function to clean up and exit
cleanup_and_exit() {
    local exit_code=$1
    echo "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    exit "$exit_code"
}

# Function to handle errors
handle_error() {
    echo "Error occurred in script at line $1"
    cleanup_and_exit 1
}

# Function to display error message and exit
display_error_and_exit() {
    echo "====================================================================="
    echo "ERROR: $1"
    echo ""
    echo "SOLUTION: $2"
    echo "====================================================================="
    cleanup_and_exit 1
}

# Set up error handling
trap 'handle_error $LINENO' ERR

# Main Google Drive folder ID
MAIN_FOLDER_ID="1k-amlOwNQEWGx751MtWZc4SbZCUs8iqK"
TEMP_DIR="temp_download"

echo "DrugGEN Setup Script"
echo "===================="
echo "This script will download datasets, encoders/decoders, model weights"
echo "and SMILES correction files from Google Drive repository."
echo ""

# Install required packages if not already installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown package for Google Drive downloads..."
    pip install gdown || cleanup_and_exit 1
fi

if ! command -v curl &> /dev/null; then
    echo "Installing curl as a backup download method..."
    apt-get update && apt-get install -y curl || cleanup_and_exit 1
fi

# Create temporary directory
rm -rf "$TEMP_DIR"  # Clean up any existing temp directory
mkdir -p "$TEMP_DIR" || cleanup_and_exit 1

echo "Downloading resources from Google Drive..."
echo "This may take some time depending on your internet connection."

# Attempt to download with gdown with more robust error handling
echo "Attempting primary download method with gdown..."
if ! gdown --folder --continue --id $MAIN_FOLDER_ID --output $TEMP_DIR; then
    echo "Initial download attempt failed. Trying alternate gdown method..."
    if ! gdown --folder --continue --id $MAIN_FOLDER_ID --output $TEMP_DIR --remaining-ok; then
        display_error_and_exit "Automated download failed. This is often due to Google Drive limitations or permission settings." "Please download the files manually:
        
1. Visit this link in your browser:
   https://drive.google.com/drive/folders/1k-amlOwNQEWGx751MtWZc4SbZCUs8iqK

2. Download the entire folder structure:
   - Download 'data/' directory and place it in the project root
   - Download 'experiments/' directory and place it in the project root
   
3. Ensure you maintain the exact directory structure from Google Drive:
   - data/encoders/ should contain encoder files
   - data/decoders/ should contain decoder files
   - experiments/models/ should contain model weights"
    fi
fi

# Verify that critical folders were downloaded before proceeding
echo "Verifying downloaded files..."

# Check for expected top-level directories
if [ ! -d "$TEMP_DIR/data" ] || [ ! -d "$TEMP_DIR/experiments" ]; then
    display_error_and_exit "Download appears to be incomplete or incorrect." "The downloaded content doesn't contain expected directories (data and experiments).
    
Please download the files manually:
1. Visit https://drive.google.com/drive/folders/1k-amlOwNQEWGx751MtWZc4SbZCUs8iqK
2. Download both the 'data/' and 'experiments/' directories
3. Place them in the project root directory, maintaining the exact same structure"
fi

# Now that we've verified the download contains the expected structure,
# we can copy all the files to their proper locations

# Create necessary directories if they don't exist
mkdir -p data/encoders data/decoders experiments/models || cleanup_and_exit 1

echo "Organizing downloaded files..."

# Define a function to copy directories recursively, preserving structure
copy_directory_contents() {
    local src_dir="$1"
    local dst_dir="$2"
    
    echo "Copying contents from $src_dir to $dst_dir..."
    
    # Create destination directory if it doesn't exist
    mkdir -p "$dst_dir"
    
    # Copy all files from source to destination
    find "$src_dir" -type f | while read file; do
        # Get the relative path from source directory
        rel_path="${file#$src_dir/}"
        dst_file="$dst_dir/$rel_path"
        dst_path=$(dirname "$dst_file")
        
        # Create destination directory if needed
        if [ ! -d "$dst_path" ]; then
            mkdir -p "$dst_path"
        fi
        
        # Copy the file
        echo "  - Copying: $rel_path"
        cp "$file" "$dst_file" || echo "Warning: Could not copy file $rel_path"
    done
}

# Copy data directory
if [ -d "$TEMP_DIR/data" ]; then
    copy_directory_contents "$TEMP_DIR/data" "data"
fi

# Copy experiments directory
if [ -d "$TEMP_DIR/experiments" ]; then
    copy_directory_contents "$TEMP_DIR/experiments" "experiments"
fi

# Verify that critical files were copied to their destinations
echo "Verifying file organization..."
MISSING_FILES=0

# Check for encoder files
if [ ! "$(ls -A data/encoders 2>/dev/null)" ] || [ "$(ls -A data/encoders 2>/dev/null | grep -v .gitkeep)" = "" ]; then
    echo "ERROR: No encoder files found in data/encoders/"
    MISSING_FILES=1
fi

# Check for decoder files
if [ ! "$(ls -A data/decoders 2>/dev/null)" ] || [ "$(ls -A data/decoders 2>/dev/null | grep -v .gitkeep)" = "" ]; then
    echo "ERROR: No decoder files found in data/decoders/"
    MISSING_FILES=1
fi

# Check for model weights
if [ ! "$(find experiments/models -type f -name "*.ckpt" -o -name "*.pt" -o -name "*.pth" 2>/dev/null)" ]; then
    echo "ERROR: No model weight files found in experiments/models/"
    MISSING_FILES=1
fi

if [ $MISSING_FILES -eq 1 ]; then
    display_error_and_exit "Failed to organize critical files properly." "Some essential files could not be copied to their appropriate destinations.
    
Please download the files manually:
1. Visit https://drive.google.com/drive/folders/1k-amlOwNQEWGx751MtWZc4SbZCUs8iqK
2. Download both the 'data/' and 'experiments/' directories 
3. Place them in the project root directory, maintaining the following structure:
   - data/encoders/ - should contain encoder files
   - data/decoders/ - should contain decoder files
   - experiments/models/ - should contain model weight files"
fi

# Print final directory structure for verification
echo ""
echo "=== Setup Complete ==="
echo "All resources have been successfully downloaded and organized."
echo "You can now train or run inference with DrugGEN."
echo ""

# Cleanup temporary files
cleanup_and_exit 0