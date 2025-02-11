for begin in 0 5000
do
for n_needles in 2 5
do
for n_row in 2 4 8
do 
    export MODEL_PROVIDER='cogvlm' MODEL_VERSION='base' N_ROW=$n_row N_COL=$n_row SEQ_LENGTH=1 N_NEEDLES=$n_needles BEGIN=$begin N_SEQ=100; 
    CUDA_VISIBLE_DEVICES=4 ECHO "YOUR COMMAND HERE."
done
done
done