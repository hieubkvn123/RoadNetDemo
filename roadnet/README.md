### How to run training script ###
```console 
	# Define the roadnet dataset directory in environment variable
	$ export ROADNET_DATADIR=<path_to_roadnet_datadir>
	$ python3 train.py --epochs <num_epochs> \
										 --batch_size <batch_size> \
										 --save_steps <save_steps> # how often to checkpoint \
									   --vis_steps <vis_steps> # how often to visualize result \
										 --learning_rate <lr> \
										 --checkpoint_path <model_checkpoint_file_path>
```

### TODO ###
	- The loss sometimes reduce to negative (suggesting there is problem with the output). Especially
		in the centerline module.
	- Add function in Trainer class to visualize result after certain steps interval in form of gif.
