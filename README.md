# graph_convolution
Graph convolution example in Keras

This is a graph convolution implementation in Keras.
Unlike CNN that does convolution among input sources regardless they are related or not, graph convolution only does convolution over related input sources.
Relation between input sources are indicated by a square neighbourhood matrix, where 1 indicates they are related, 0 otherwise.

In example, we are trying to estimate average speed of a road section 15 minutes, 30 minutes and 45 minutes in advance.
Input source is speed information of 35 road sections surrounding the target road section. 
Input matrix is a 16x35 matrix in which rows are measurement times in 15 minutes intervals and columns are input sources (traffic speed).
Output matrix is a 1x3 vector, speed of target road section 15min, 30min and 45min after.
This means we are using past 4 hours information to forecast traffic speed of 45 minutes later. 

Example data contains 671 samples, in which first 500 are used for training and remaing are used for validation.
