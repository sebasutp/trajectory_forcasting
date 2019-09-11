Trajectory Forecasting using Deep Conditional Generative Models
===============================================================

This repository provides an implementation of a method for trajectory
prediction based on Conditional Variational Auto-Encoders (CVAE).
The paper explaining all the details about this method can be found
[here](https://arxiv.org/abs/1909.03895).


Getting Started
---------------

You will need Python3 to be able to use this code. The Python module
can be installed like any other Python module using:

```
python setup.py install
```

After the module is installed, you can run the example scripts located
on the "examples/" folder. These example scripts can be run first with
the "--help" argument on the command line to obtain a small description
of how to run the script and what parameters are required.


Training a ball model
---------------------

One of the examples you can run consists on training a model to predict
table tennis ball trajectories. To generate a simulated ball data set,
you can run

```
python examples/gen_sim_ball_dataset.py /tmp/sim_data.npz 2000 --T 200 --max_bounces 1
```

generating 2000 simulated ball trajectories in the file 
"/tmp/sim_data.npz", each of length of 200 ball trajectories, and 
bouncing at most once on the table.

Subsequently, we can train a ball model using the trajectory variational
auto-encoder using for example 400 epochs, a batch size of 128 trajectories
and 5% of data for the validation set.

```
python examples/train_dcgm.py /tmp/sim_data.npz /tmp/sim_model --p 0.05 --epochs 400 --batch_size 128
```

The previous script has options to generate missing observations and
outliers during training to make the model more robust for prediction.
After the model is trained, we can plot for example the predictions
for the first 10 trajectories of the training set using

```
python examples/plot_dist.py /tmp/sim_data.npz /tmp/sim_model --n 10 --t 1.0
```

You can generate a different set of data (a test set) to assess the 
generalization quality using the same command.

Model API
---------

To create a trajectory variational auto-encoder use the "TrajDCGM" class.

```
import traj_pred.dcgm as dcgm

# create the following Keras models with the architecture you need:
# encoder = keras.models.Model(inputs=[x,x_obs], outputs=[mu_z, log_sig_z])
# decoder = keras.models.Model(inputs=[x,x_obs,z], outputs=[mu_y])

model = dcgm.TrajDCGM(encoder=encoder, partial_encoder=encoder, cond_generator=decoder, log_sig_y=log_sig_y, length=length, D=D, z_size=z_size)

# Then train it using the regular Keras API using data Generators
# Check the "fit_generator" API documentation of Keras
model.fit_generator(training_set, test_set, epochs=epochs)
```

The previous code skeleton should be relatively easy to understand after
reading the paper. You can decide the architecture of the model you want
to use for the encoder and decoder using the Keras model object.
