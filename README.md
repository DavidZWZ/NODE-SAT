## NODE-SAT
## Environments

[PyTorch 1.8.1](https://pytorch.org/),
[numpy](https://github.com/numpy/numpy),
[pandas](https://github.com/pandas-dev/pandas),
[tqdm](https://github.com/tqdm/tqdm),
[tabulate](https://github.com/astanin/python-tabulate),
[torchdiffeq](https://github.com/rtqichen/torchdiffeq)

## Datasets
The datasets can be downloaded [here](https://zenodo.org/record/7213796#.Y1cO6y8r30o). 
Please download them and put them in ```DG_data``` folder. 
Thirteen datasets are used in NODE-SAT:
- Wikipedia
- Reddit
- MOOC
- LastFM
- Enron
- Social Evo.
- UCI
- Flights
- Can. Parl.
- US Legis.
- UN Trade
- UN Vote
- Contact

Run ```preprocess_data/preprocess_data.py``` for pre-processing the datasets.
For example, to preprocess the *Wikipedia* dataset, we can run the following commands:
```{bash}
cd preprocess_data/
python preprocess_data.py  --dataset_name wikipedia
```
We can also run the following commands to preprocess all the original datasets at once:
```{bash}
cd preprocess_data/
python preprocess_all_data.py
```

## Executing
### Training
* Example of training *NODE-SAT*:
```bash
# Contacts dataset
python train_link_prediction.py --dataset_name Contacts --model_name NODE-SAT --max_input_sequence_length 64 --num_runs 5 --gpu 0 --num_layers 1  --batch_size 200 --negative_sample_strategy random --num_epochs 5 --num_heads 1 --num_neighbors 50 --time_param 1

# Enron dataset
python train_link_prediction.py --dataset_name enron --model_name NODE-SAT --max_input_sequence_length 64 --num_runs 5 --gpu 0 --num_layers 4  --batch_size 200 --negative_sample_strategy random --num_epochs 5 --num_heads 1 --num_neighbors 20 --time_param 1.3

# Flights dataset
python train_link_prediction.py --dataset_name Flights --model_name NODE-SAT --max_input_sequence_length 64 --num_runs 5 --gpu 0 --num_layers 1 --batch_size 200 --negative_sample_strategy random --num_epochs 4 --num_heads 1 --num_neighbors 50 --time_param 1

# Last.fm dataset
python train_link_prediction.py --dataset_name lastfm --model_name NODE-SAT --max_input_sequence_length 64 --num_runs 5 --gpu 0 --num_layers 3  --batch_size 200 --negative_sample_strategy random --num_epochs 10 --num_heads 1 --num_neighbors 50 --time_param 1

# MOOC dataset
python train_link_prediction.py --dataset_name mooc --model_name NODE-SAT --max_input_sequence_length 64 --num_runs 5 --gpu 0 --num_layers 1  --batch_size 200 --negative_sample_strategy random --num_epochs 10 --num_heads 1 --num_neighbors 50 --time_param 1

# Reddit dataset
python train_link_prediction.py --dataset_name reddit --model_name NODE-SAT --max_input_sequence_length 64 --num_runs 5 --gpu 0 --num_layers 1  --batch_size 200 --negative_sample_strategy random --num_epochs 5 --num_heads 1 --num_neighbors 50 --time_param 1

# SocialEvo dataset
python train_link_prediction.py --dataset_name SocialEvo --model_name NODE-SAT --max_input_sequence_length 64 --num_runs 5 --gpu 0 --num_layers 1  --batch_size 200 --negative_sample_strategy random --num_epochs 5 --num_heads 1 --num_neighbors 50 --time_param 1

# UCI dataset
python train_link_prediction.py --dataset_name uci --model_name NODE-SAT --max_input_sequence_length 64 --num_runs 5 --gpu 0 --num_layers 1  --batch_size 200 --negative_sample_strategy random --num_epochs 8 --num_heads 1 --num_neighbors 50 --time_param 1

# UN Trade dataset
python train_link_prediction.py --dataset_name UNtrade --model_name NODE-SAT --max_input_sequence_length 64 --num_runs 5 --gpu 0 --num_layers 3  --batch_size 200 --negative_sample_strategy random --num_epochs 10 --num_heads 1 --num_neighbors 50 --time_param 1

# UN Vote dataset
python train_link_prediction.py --dataset_name UNvote --model_name NODE-SAT --max_input_sequence_length 64 --num_runs 5 --gpu 0 --num_layers 3  --batch_size 200 --negative_sample_strategy random --num_epochs 10 --num_heads 1 --num_neighbors 50 --time_param 1

# US Legislature dataset
python train_link_prediction.py --dataset_name USLegis --model_name NODE-SAT --max_input_sequence_length 64 --num_runs 5 --gpu 0 --num_layers 1  --batch_size 200 --negative_sample_strategy random --num_epochs 10 --num_heads 1 --num_neighbors 20 --time_param 1.5

# Wikipedia dataset
python train_link_prediction.py --dataset_name wikipedia --model_name NODE-SAT --max_input_sequence_length 64 --num_runs 5 --gpu 0 --num_layers 1  --batch_size 200 --negative_sample_strategy random --num_epochs 5 --num_heads 1 --num_neighbors 20 --time_param 1

# Canadian Parliament dataset
python train_link_prediction.py --dataset_name CanParl --model_name NODE-SAT --max_input_sequence_length 64 --num_runs 5 --gpu 0 --num_layers 2  --batch_size 200 --negative_sample_strategy random --num_epochs 5 --num_heads 1 --num_neighbors 20 --time_param 1
```