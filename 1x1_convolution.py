
# coding: utf-8




from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)





import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10])) #0에서부터 9까지 총 10개
b = tf.Variable(tf.zeros([10]))
#sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables())
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



'''
def tf_get_variable("name_initial",shape):
  
  name_initial = tf.get_variable("name_initial",shape, #절단 전의 정규분포의 표준편차는 0.1입니다. 
  intializer= tf.contrib.layers.xavier_initializer())
  return tf.Variable(name_initial) #변수?함수밖에서 사용가능한 변수로 만들어준다??
'''
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) #절단 전의 정규분포의 표준편차는 0.1입니다. 
  return tf.Variable(initial) #변수?함수밖에서 사용가능한 변수로 만들어준다??

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape) #0.1값을 정해진 규격 배열?을 채운다.
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #x와 W만 입력받아도 자동으로
                                                                  #strides와 padding을 생성

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], #[batch ,높이 ,너비, RGB(그레이스케일이라 1?)]
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], #[batch ,높이 ,너비, RGB(그레이스케일이라 1?)]
                        strides=[1, 3, 3, 1], padding='SAME')




#W_conv1 = weight_variable([5, 5, 1,32]) #4차원 배열을 넣어준거지,넣어준 값과 표준편차 0.1 차이로??
                                       #계속 생성?? 임의 느낌??[필터사이즈(2),채널, 커널(출력채널수)]
W_conv1 = tf.get_variable("W_conv1",shape=[1, 1, 1,8],initializer=tf.contrib.layers.xavier_initializer())
b_conv1 = bias_variable([8]) # 0.1값을 (1*32)1차원배열에 넣는것?
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_conv1 = tf.nn.sigmoid(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
print(h_pool1)
W_conv2 = tf.get_variable("W_conv2",shape= [5, 5, 8,32],initializer=tf.contrib.layers.xavier_initializer())
#W_conv2 = weight_variable([5, 5, 32, 64])#위의 전 컨볼루션 출력이 32이였으니 32를 그대로 받고 64개 출력한다?
#W_conv2 = weight_variable([5, 5, 32, 32]) #커널*채널값이 커널이 된다? 적용하기위해 커널값은 고정을 시켜봄
b_conv2 = bias_variable([32]) #커널 갯수를 따라 맞춘다?
#b_conv2 = bias_variable([1024])#32*32

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
print(h_pool2)
# 3번 convolution 추가한것 
#W_conv3 = weight_variable([5, 5, 64, 128])
#W_conv3 = weight_variable([5, 5, 1024, 32])
W_conv3 = tf.get_variable("W_conv3",shape= [5, 5, 32,64],initializer=tf.contrib.layers.xavier_initializer())
b_conv3 = bias_variable([64])
#b_conv3 = bias_variable([32768])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#h_conv3 = tf.nn.sigmoid(conv2d(h_pool2, W_conv3) + b_conv3)

#h_pool3 = max_pool_2x2(h_conv3)
#print(h_pool3)
#print(h_pool3)
# 여기까지

W_fc1 = weight_variable([7 * 7 * 64, 1024])
#W_fc1 = weight_variable([7 * 7 * 128, 1024])#1024면 괜찮더라? 이런 거임 걍쓰는 숫자
b_fc1 = bias_variable([1024])

''' fc추가를 위해서 주석처리 했음 변경해야하는데 원본 보존
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
'''
#h_pool3_flat = tf.reshape(h_pool3, [-1, 7*7*128])
h_pool3_flat = tf.reshape(h_conv3, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
#h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


#W_fc3 = weight_variable([7 * 7 * 64, 1024]) # I forgive fc_3

#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)






cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# tf.reduce_sum; 벡터의 행과 열끼리 합쳐서 하나로 뭉침?
# tf.reduce_mean; 벡터의 행과 열끼리 평균냄. 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(1000):
  batch = mnist.train.next_batch(64)#한묶음에 64개씩 처리?
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

