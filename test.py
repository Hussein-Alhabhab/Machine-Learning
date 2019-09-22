import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#set execution engine
tf.enable_eager_execution()

#plot
x_data= np.load('x_hw.npy')
y_data= np.load('y_hw.npy')
#plt.plot(x_data, y_data)
#plt.show()


#create variables
a1 = tf.Variable(tf.ones([1]))
a2 = tf.Variable(tf.ones([1]))
f1 = tf.Variable(tf.ones([1]))
f2 = tf.Variable(tf.ones([1]))

# create model 

y = a1*np.sin(f1*x_data) + a2*np.sin(f2*x_data) 

# loss 
loss = tf.reduce_mean(tf.square(y-y_data))
# optimizer
optimizer =  tf.compat.v1.train.AdamOptimizer(0.10)
train_step = optimizer.minimize(loss)

##########################################################################################################

session = tf.InteractiveSession()
#tf.global_variables_initializer().run()

#sav

N = 10000 
for x in range(N):
	session.run(train_step)
	if x%200 == 0:
		print("x=",x,"a1=", session.run(a1),"a2=", session.run(a2),"f1=",session.run(f1),"f2=",session.run(f2),"loss=",session.run(loss))

A1,A2,F1,F2 = session.run([a1,a2,f1,f2])
A1=A1[0]
A2=A2[0]
F1=F1[0]
F2=F2[0]

print(A1,A2,F1,F2)

print('Success')

