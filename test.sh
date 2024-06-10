#!/bin/bash

# Define arrays of values for N_ROW, N_COL, and SEQ_LENGTH
n_values=(1 2 4 8)
seq_lengths=(1 10)

# Loop over each combination of N_ROW, N_COL, and SEQ_LENGTH
for seq_len in "${seq_lengths[@]}"; do
    for n in "${n_values[@]}"; do
        # Check if both N_ROW and N_COL and SEQ_LENGTH are not all 1
        if ! [ "$n" -eq 1 -a "$seq_len" -eq 1 ]; then
            export N_ROW=$n
            export N_COL=$n
            export SEQ_LENGTH=$seq_len

            # Execute the Python script with the current environment settings
            python needle.py
        fi
    done
done
