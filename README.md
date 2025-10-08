# MAMBAINR

Follows the trans-inr repo structure, which can be found at https://github.com/yinboc/trans-inr

You can run the example files with the following command from trans-inr-master folder:

bash trainer_fMRI_ddp.sh example.yaml hcp_data_all_config.yaml
- this will create a log in trans-inr-master/save

This example reproduces the experiment for the 4^3 patch size, 36^3 volume size experiment from the 4D fMRI volume table in the workshop paper. 

The current Mamba-GINR model is defined in trans-inr-master/models/mamba_lainr_inr_ddp.py
The decoder hyponet component used is trans-inr-master/models/hyponet/lainr_mlp_bias_fmri.py 
