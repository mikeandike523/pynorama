#!/bin/bash

rm  -rf "./testing/output"
mkdir "./testing/output"

# Define the input and output directories
input_dir="./testing/resources/test-sets"
output_dir="./testing/output/test-sets"

# Check if the input directory exists
if [ ! -d "$input_dir" ]; then
    echo "Input directory $input_dir does not exist."
    exit 1
fi

rm -rf "$output_dir"
mkdir -p "$output_dir"

# Iterate over each directory in the input directory
for dir in "$input_dir"/*; do
    if [ -d "$dir" ]; then
        # Get the basename of the directory
        dir_basename=$(basename "$dir")

        # Define the output path for the current directory
        output_path="$output_dir/$dir_basename"

        # Run the command
        ./pynorama "./testing/resources/test-sets/$dir_basename" "$output_path.png"

        # Check if the command was successful
        if [ $? -ne 0 ]; then
            echo "Error running ./pynorama for $dir_basename"
        else
            echo "Successfully processed $dir_basename"
        fi
    fi
done
