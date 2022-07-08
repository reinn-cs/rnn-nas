# Multi-Objective Evolutionary Neural Architecture Search for Recurrent Neural Networks

This repo contains the source code for the multi-objective evolutionary algorithm based recurrent neural network architecture search implementation done for the Master's thesis titled 'Multi-Objective Evolutionary Neural Architecture Search for Evolving Recurrent Neural Networks'.

This implementation is done on top of the PyTorch framework. 

The default env configuration (config/env.json) can be used as-is to search for RNN architectures for the PTB dataset. This will consider a population size of 100 architectures and generate 100 offspring architectures for each generation. 

To execute the search, simply run the following: `python run_ptb_search.py`

Before running, please run `pip install -r requirements.txt` - it is recommended to create an isolated virtual environment first. All experiments were run using Anaconda for dependency management.


The search restore checkpoints for experiments done for the MSc dissertation can be provided on request (the files are quite large and in excess of tens of GBs).
