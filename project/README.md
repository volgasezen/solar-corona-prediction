This repository is very WIP. 

## Folder Structure

* 0-cnn_lstm_cnn: First version trained end-to-end.
* 1-cnn_ae: Where many CNN based autoencoders are trained and tested.
* 2-conv_lstm: Despite the name, this is where the flat LSTM was tarined and tested.
* 3-lstm2: Convolutional LSTM training, testing, checkpoints, and logs. (Autoencoders here supply 16 filters, whereas those in the root supply 128 filters.)
* 4-phydnet: Early tests of phydnet, did not make it to the project report.

While the project requires a .zarr file downloaded from AWS, the val_x, val_y npy files provide the necessary information to compute the performance metrics.

Sorry for the mess, housekeeping is scheduled.
