@REM # #!/bin/bash

@REM # # Define arrays of values for N_ROW, N_COL, and SEQ_LENGTH
@REM # n_values=(1 2 4 8)
@REM # seq_lengths=(1 10)

@REM # # Loop over each combination of N_ROW, N_COL, and SEQ_LENGTH
@REM # for seq_len in "${seq_lengths[@]}"; do
@REM #     for n in "${n_values[@]}"; do
@REM #         # Check if both N_ROW and N_COL and SEQ_LENGTH are not all 1
@REM #         if ! [ "$n" -eq 1 -a "$seq_len" -eq 1 ]; then
@REM #             export N_ROW=$n
@REM #             export N_COL=$n
@REM #             export SEQ_LENGTH=$seq_len
@REM #             echo "Running with N_ROW=$N_ROW, N_COL=$N_COL, SEQ_LENGTH=$SEQ_LENGTH"
@REM #             # Execute the Python script with the current environment settings
@REM #             python3 custom_needle.py
@REM #         fi
@REM #     done
@REM # done

@echo off

:: Define arrays of values for N_ROW, N_COL, and SEQ_LENGTH
set n_values=1 2 4 8
set seq_lengths=1 10

:: Loop over each combination of N_ROW, N_COL, and SEQ_LENGTH
for %%S in (%seq_lengths%) do (
    for %%N in (%n_values%) do (
        :: Check if both N_ROW and N_COL and SEQ_LENGTH are not all 1
        if not "%%N"=="1" if not "%%S"=="1" (
            set N_ROW=%%N
            set N_COL=%%N
            set SEQ_LENGTH=%%S
            echo Running with N_ROW=%N_ROW%, N_COL=%N_COL%, SEQ_LENGTH=%SEQ_LENGTH%
            :: Execute the Python script with the current environment settings
            python custom_needle.py
        )
    )
)