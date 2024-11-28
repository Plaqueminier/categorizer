#!/bin/bash

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 source_dir destination_dir"
    exit 1
fi

source_dir="$1"
dest_dir="$2"

# Check directories
if [ ! -d "$source_dir" ] || [ ! -d "$dest_dir" ]; then
    echo "Both source and destination directories must exist"
    exit 1
fi

# Count total files
total_files=$(ls -1 "$source_dir"/*.{jpg,jpeg,png,gif} 2>/dev/null | wc -l)

# Calculate 10%
files_to_move=$(( (total_files + 9) / 10 ))

if [ $total_files -eq 0 ]; then
    echo "No image files found"
    exit 1
fi

# Get random files using sort -R
ls "$source_dir"/*.{jpg,jpeg,png,gif} 2>/dev/null | sort -R | head -n $files_to_move | while read file; do
    mv "$file" "$dest_dir"
    echo "Moved: $file"
done

echo "Moved $files_to_move files out of $total_files total files"