# IADB_Diffusion_Model_Optimization

The aim of this project is to further optimize the Iterative De-Blending Diffusion model in terms of training speed. To do so, the optimizer used to update the weights of the model was altered from the AdamW (as in the original paper) to SGD for experimentation purposes. More optimizers will be used for testing in the future.

As of now, the AdamW optimizer is the faster algorithm, as it reached a loss of around 100000 at around timestep 800 (and asymptotically reached a loss of 60000) while the SGD optimizer has yet to reach these losses as shown in the graph. Additionally, the learning rate used for SGD was 1e-8, as higher rates (such as 1e-4 and 1e-6) resulted in progressively increasing error. More experimentation is being done to identify an ideal rate.

Resources used: https://github.com/tchambon/IADB
