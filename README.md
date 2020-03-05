# Capacity Estimation via Directed Information Neural Estimator (DINE)

This repository contains an implementation of capacity estimator of continuous channels as introduced in (Link will be added in the future).

## Prerequisites

The code is compatible with a tensorflow 2.0 environment.
If you use a docker, you can pull the following docker image

```
docker pull tensorflow/tensorflow:latest-gpu-py3
```


## Running the code

The parameters of the channels are in the file `configs.py`. Modify this file to set your own values for the parameters (for example, the transition matrix or the noise covariance matrix for the MIMO channel.

The estimate the capacity of the AWGN channel run
```
python ./main.py --name <simulation_name> --config_name awgn --P <source_power> &
```
The estimate the capacity of the MA(1)-AGN channel run
```
python ./main.py --name <simulation_name> --config_name arma_ff --P <source_power> &
```
The estimate the feedback capacity of the MA(1)-AGN channel run
```
python ./main.py --name <simulation_name> --config_name arma_fb --P <source_power> &
```
The estimate the capacity of the MIMO channel, set the desired parameters in the file `configs.py` and run
```
python ./main.py --name <simulation_name> --config_name mimo --P <source_power> &
```
## Authors

* **Ziv Aharoni** 
* **Dor Tsur** 
* **Ziv Goldfeld** 
* **Haim Permuter** 
* **Javier Garcia Gomez** 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

