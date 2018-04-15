#!/bin/bash
#SBATCH --mem=40GB
#SBATCH --job-name=2-gpu-bidaf-pytorch
#SBATCH --output=run_logs/res_%j.txt 	# output file
#SBATCH -e run_logs/res_%j.err        	# File to which STDERR will be written
#
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shasvatmukes@cs.umass.edu
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python/3.6.1
module load cuda80/blas/8.0.44
module load cuda80/fft/8.0.44
module load cuda80/nsight/8.0.44
module load cuda80/profiler/8.0.44
module load cuda80/toolkit/8.0.44

## Change this line so that it points to your bidaf github folder

# Training (Default - on SQuAD)
python -m allennlp.run train training_config/bidaf__serialization_dir__multihead_2.json -s output_path_multi_head_2
# Evaluation (Default - on SQuAD)
python -m allennlp.run evaluate https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz --evaluation-data-file data/dev.json
# Evaluate on NewsQA
#python -m allennlp.run evaluate https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz --evaluation-data-file "data/newsqa_raw/test-v1.1.json"

# Prediction on a user defined passage
#echo '{"passage": "A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight.", "question": "How many partially reusable launch systems were developed?"}'

echo '{"passage":"There were three friends sitting on three chairs and talking about deep learning. One of them said I am hungry. So the other one gave him three apples.", "question": "How many people were sitting on the chairs?"}' > examples.jsonl
python -m allennlp.run predict https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz  examples.jsonl
