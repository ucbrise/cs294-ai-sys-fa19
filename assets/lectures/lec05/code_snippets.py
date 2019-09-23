



import numpy
import theano.tensor as T
from theano import shared, function

x = T.matrix()
y = T.lvector()
w = shared(numpy.random.randn(100))
b = shared(numpy.zeros(()))

print "Initial model:"
print w.get_value(), b.get_value()

p_1 = 1 / (1 + T.exp(-T.dot(x, w)-b))
xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1)
cost = xent.mean() + 0.01*(w**2).sum()
gw,gb = T.grad(cost, [w,b])
prediction = p_1 > 0.5


predict = function(inputs=[x],
                   outputs=prediction)
train = function(
    inputs=[x,y],
    outputs=[prediction, xent],
    updates={w: w - 0.1*gw, b: b - 0.1*gb})

N = 4
feats = 100
D = (numpy.random.randn(N, feats),
numpy.random.randint(size=N,low=0, high=2))
training_steps = 10
for i in range(training_steps):
	pred, err = train(D[0], D[1])
print "Final model:",
print w.get_value(), b.get_value()
print "target values for D", D[1]
print "prediction on D", predict(D[0])



tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])




from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



steps = [('scaler', StandardScaler()), ('SVM', SVC())]
pipeline = Pipeline(steps) # define the pipeline object.
parameteres = {'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01]}
grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5)
grid.fit(X_train, y_train)



c = []
for d in ['/device:GPU:2', '/device:GPU:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))




    if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!




    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    print(y)

    # tensor([[3., 3.],
    #         [3., 3.]], grad_fn=<AddBackward0>)

    z = y * y * 3
    out = z.mean()

    print(out)

    # tensor(27., grad_fn=<MeanBackward0>)


    out.backward()
    print(x.grad)

    # tensor([[4.5000, 4.5000],
    #         [4.5000, 4.5000]])



    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    print(y)
    z = y * y * 3
    out = z.mean()
    print(out)
    out.backward()
    print(x.grad)




# Define the model sequentially
model = tf.keras.Sequential([
    # Adds two densely-connected layers with 64 units:
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(64, activation='relu'),
    # Add a softmax layer with 10 output units:
    layers.Dense(10, activation='softmax')])

# Setup the model and training routines
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Invent some Data
data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))
# Train the model
model.fit(data, labels, epochs=10, batch_size=32)
# Make predictions
result = model.predict(data, batch_size=32)


import tensorflow as tf

tf.enable_eager_execution()



a = tf.constant([[1, 2],
                 [3, 4]])
b = (a + 1) * (a - 1)

# array([[ 0,  3],
#        [ 8, 15]], dtype=int32)>




x = tf.Variable(tf.ones([2,2]))
with tf.GradientTape() as tape:
  y = x + 2
  z = y * y * 3
  out = tf.math.reduce_mean(z)
  print(out)
grad = tape.gradient(out, x)
print(grad)


