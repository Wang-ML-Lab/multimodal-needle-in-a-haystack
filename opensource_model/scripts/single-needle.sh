for begin in 0 5000
do
for n_needles in 1
do
for n_row in 2 4 8
do 
    export MODEL_PROVIDER='cogvlm' MODEL_VERSION='base' N_ROW=$n_row N_COL=$n_row SEQ_LENGTH=1 N_NEEDLES=$n_needles BEGIN=$begin N_SEQ=1000; 
    CUDA_VISIBLE_DEVICES=5 ECHO "YOUR COMMAND HERE."
done
done
done