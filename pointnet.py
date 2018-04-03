import os
import h5py
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf

##### helper functions #####

def load_h5(h5_filename):
    """
    Data loader function.
    Input: The path of h5 filename
    Output: A tuple of (data,label)
    """
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def get_category_names():
    """
    Function to list out all the categories in MODELNET40
    """
    shape_names_file = os.path.join('', 'shape_names.txt')
    shape_names = [line.rstrip() for line in open(shape_names_file)]
    return shape_names

############################

# read in data
train0 = load_h5('ply_data_train0.h5')
train1 = load_h5('ply_data_train1.h5')
train2 = load_h5('ply_data_train2.h5')
train3 = load_h5('ply_data_train3.h5')
train4 = load_h5('ply_data_train4.h5')

test0 = load_h5('ply_data_test0.h5')
test1 = load_h5('ply_data_test1.h5')

N_train = np.zeros(5)
N_train[0] = train0[1].shape[0]
N_train[1] = train1[1].shape[0]
N_train[2] = train2[1].shape[0]
N_train[3] = train3[1].shape[0]
N_train[4] = train4[1].shape[0]

N_test = np.zeros(2)
N_test[0] = test0[1].shape[0]
N_test[1] = test1[1].shape[0]

train_data = np.zeros([int(np.sum(N_train)), 2048, 3])
idx = 0
train_data[idx:idx+int(N_train[0]),:,:] = train0[0]
idx += int(N_train[0])
train_data[idx:idx+int(N_train[1]),:,:] = train1[0]
idx += int(N_train[1])
train_data[idx:idx+int(N_train[2]),:,:] = train2[0]
idx += int(N_train[2])
train_data[idx:idx+int(N_train[3]),:,:] = train3[0]
idx += int(N_train[3])
train_data[idx:idx+int(N_train[4]),:,:] = train4[0]

train_labels = np.zeros([int(np.sum(N_train)), 1])
idx = 0
train_labels[idx:idx+int(N_train[0]),:] = train0[1]
idx += int(N_train[0])
train_labels[idx:idx+int(N_train[1]),:] = train1[1]
idx += int(N_train[1])
train_labels[idx:idx+int(N_train[2]),:] = train2[1]
idx += int(N_train[2])
train_labels[idx:idx+int(N_train[3]),:] = train3[1]
idx += int(N_train[3])
train_labels[idx:idx+int(N_train[4]),:] = train4[1]

test_data = np.zeros([int(np.sum(N_test)), 2048, 3])
idx = 0
test_data[idx:idx+int(N_test[0]),:,:] = test0[0]
idx += int(N_test[0])
test_data[idx:idx+int(N_test[1]),:,:] = test1[0]

test_labels = np.zeros([int(np.sum(N_test)), 1])
idx = 0
test_labels[idx:idx+int(N_test[0]),:] = test0[1]
idx += int(N_test[0])
test_labels[idx:idx+int(N_test[1]),:] = test1[1]

train_one_hot_labels = np.zeros([int(np.sum(N_train)),40])
for idx, label in enumerate(train_labels.reshape(int(np.sum(N_train)),).tolist()):
	train_one_hot_labels[idx, int(label)] = 1

test_one_hot_labels = np.zeros([int(np.sum(N_test)),40])
for idx, label in enumerate(test_labels.reshape(int(np.sum(N_test)),).tolist()):
	test_one_hot_labels[idx, int(label)] = 1

ix_to_shape = {ix:shape for ix,shape in enumerate(get_category_names())}

# rotate point clouds
def rotate_points(points): 
	B = points.shape[0]
	N = points.shape[1]

	# unifrom sample angles for rotation
	phi = np.random.uniform(0, 2*np.pi,B)

	rotated_points = np.zeros([B,N,3])
	for i in range(B):
		cos_ = np.cos(phi[i])
		sin_ = np.sin(phi[i])
		rotZ = np.array([[cos_, 0, sin_],\
		                 [0, 1, 0],\
		                 [-sin_, 0, cos_]])
		rotated_points[i,:,:] = np.dot(rotZ, points[i,:,:].T).T

	return rotated_points

# jitter point clouds
def jitter_points(points, std=0.01): 
	# add random noise
    jittered_points = std*np.random.randn(points.shape[0], points.shape[1], 3)
    jittered_points += points

    return jittered_points

# input and feature transform based on parameter k
def get_transform(points, k=3, reuse=False):
	transform = tf.layers.conv1d(points,64,1,activation=tf.nn.relu)
	transform = tf.contrib.layers.batch_norm(transform)
	transform = tf.layers.conv1d(transform,128,1,activation=tf.nn.relu)
	transform = tf.contrib.layers.batch_norm(transform)
	transform = tf.layers.conv1d(transform,1024,1,activation=tf.nn.relu)
	transform = tf.contrib.layers.batch_norm(transform)
	transform = tf.reduce_max(transform,axis=1)
	transform = tf.layers.dense(transform,512,activation=tf.nn.relu)
	transform = tf.contrib.layers.batch_norm(transform)
	transform = tf.layers.dense(transform,256,activation=tf.nn.relu)
	transform = tf.contrib.layers.batch_norm(transform)

	# identity = np.identity(k).astype(np.float32).reshape([1,k*k])

	# w = np.zeros([256,k*k]).astype(np.float32)
	# w[:,:] = identity
	# init_weights = tf.constant_initializer(w)

	# b = np.zeros([k*k]).astype(np.float32)
	# init_biases = tf.constant_initializer(b)
	
	weights = tf.get_variable("weights"+str(k),[256,k*k],initializer=tf.constant_initializer(0.0))
	biases = tf.get_variable("biases"+str(k),[k*k],initializer=tf.constant_initializer(0.0))

	# initialize the output matrix to identity matrix
	biases += tf.constant(np.eye(k).flatten(), dtype=tf.float32)

	transform = tf.matmul(transform,weights)
	transform = tf.nn.bias_add(transform,biases)

	transform = tf.reshape(transform,[tf.shape(inputs)[0],k,k])
	transformed_points = tf.matmul(points,transform)

	# also return the transformation matrix 
	return transform, transformed_points

###### vanilla version ######

inputs = tf.placeholder(tf.float32, [None,2048,3])
y = tf.placeholder(tf.float32, [None,40])
labels = tf.placeholder(tf.int64, [None,1])
lr = tf.placeholder(tf.float32, [])
input_size = tf.placeholder(tf.int32, [])

# input transform 
_, transform = get_transform(inputs)

mlp = tf.layers.conv1d(transform,64,1,activation=tf.nn.relu)
mlp = tf.contrib.layers.batch_norm(mlp)
mlp = tf.layers.conv1d(mlp,64,1,activation=tf.nn.relu)
mlp = tf.contrib.layers.batch_norm(mlp)

# feature transform
a, transform = get_transform(mlp,k=64,reuse=True)

mlp = tf.layers.conv1d(transform,64,1,activation=tf.nn.relu)
mlp = tf.contrib.layers.batch_norm(mlp)
mlp = tf.layers.conv1d(mlp,128,1,activation=tf.nn.relu)
mlp = tf.contrib.layers.batch_norm(mlp)
mlp = tf.layers.conv1d(mlp,1024,1,activation=tf.nn.relu)
mlp = tf.contrib.layers.batch_norm(mlp)

# max pooling
features = tf.reduce_max(mlp,axis=1)

fc = tf.layers.dense(features,512,activation=tf.nn.relu)
fc = tf.contrib.layers.batch_norm(fc)
fc = tf.layers.dense(fc,256,activation=tf.nn.relu)
fc = tf.contrib.layers.batch_norm(fc)
fc = tf.nn.dropout(fc,0.7)
fc = tf.layers.dense(fc,40)

logits = tf.nn.softmax(fc)
predictions = tf.argmax(logits,axis=1)
predictions = tf.reshape(predictions,[-1,1])

# compute the regularization term
d = tf.eye(64,batch_shape=[input_size])-tf.matmul(a,tf.transpose(a,perm=[0,2,1]))
# l_reg = tf.sqrt(tf.reduce_sum(tf.multiply(d,d)))
l_reg = tf.nn.l2_loss(d)

loss = tf.reduce_mean(\
	   tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))\
	   +0.001*l_reg

accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

#############################

learning_rate = 0.001
batch_size = 32
num_epochs = 300

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

ep = 0
epochs_loss = []
epochs_accu = []
while ep < num_epochs:
	idx = np.random.permutation(int(np.sum(N_train)))
	train_data = train_data[idx,:,:]
	train_labels = train_labels[idx,:]
	train_one_hot_labels = train_one_hot_labels[idx,:]

	# half the learning rate every 20 epochs
	if ep % 20 == 0 and ep > 0:
		learning_rate /= 2

	ep_loss = 0.
	ep_accu = 0.
	n = int(np.ceil(np.sum(N_train)/batch_size))
	for i in range(n):
		s = i*batch_size
		e = min((i+1)*batch_size,int(np.sum(N_train)))

		batch_data = train_data[s:e,:,:]
		batch_labels = train_labels[s:e,:]
		batch_one_hot_labels = train_one_hot_labels[s:e,:]

		# choose some of the points to rotate and jitter
		rotate_op = np.random.randint(10)==0
		jitter_op = np.random.randint(10)==9

		if rotate_op:
			batch_data = rotate_points(batch_data)
		if jitter_op:
			batch_data = jitter_points(batch_data)

		_, l, accu = sess.run([optimizer, loss, accuracy],\
                           feed_dict={inputs: batch_data,\
                           			  y: batch_one_hot_labels,\
                           			  labels: batch_labels.astype(np.int64),\
                           			  lr: learning_rate,\
                           			  input_size: (e-s)})

		ep_loss += l
		ep_accu += accu

	epochs_loss.append(ep_loss/n)
	epochs_accu.append(ep_accu/n)
	print("epoch={}, train loss={}".format(ep,epochs_loss[-1]))
	print("epoch={}, train accuracy={}\n".format(ep,epochs_accu[-1]))
	ep += 1

# x_axis = [i for i in range(num_epochs)]
# plt.figure
# plt.plot(x_axis, epochs_loss)
# plt.savefig('train_loss.png')
# plt.figure
# plt.plot(x_axis, epochs_accu)
# plt.savefig('train_accuracy.png')

n = int(np.ceil(np.sum(N_test)/batch_size))
test_accu = 0.
for i in range(n):
	s = i*batch_size
	e = min((i+1)*batch_size,int(np.sum(N_test)))
	batch_data = test_data[s:e,:,:]
	batch_labels = test_labels[s:e,:]
	accu = sess.run(accuracy,\
						 feed_dict={inputs: batch_data,\
				               		labels: batch_labels.astype(np.int64)})
	test_accu += accu
test_accu /= n
print('test accuracy={}'.format(test_accu)) # vanilla: 0.8713942307692307
