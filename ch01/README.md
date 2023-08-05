## 1.1.1 生成向量（张量）和矩阵

```ruby
require "numo/narray"

x = Numo::NArray[1, 2, 3]
x.class # => Numo::Int32
x.shape # => [3]
x.ndim # => 1

w = Numo::NArray[[1, 2, 3], [4, 5, 6]]
w.class # => Numo::Int32
w.shape # => [2, 3]
w.ndim # => 2

v = Numo::NArray[1.00001]
v.class # => Numo::DFloat

require "torch-rb"

Torch.from_num(w) # => tensor([[1, 2, 3], [4, 5, 6]], dtype: :int32)
```

```ruby
require "torch-rb"

x = Torch.tensor([1, 2, 3])
x.class # => Torch::Tensor
x.shape # => [3]
x.ndim # => 1

w = Torch.tensor([[1, 2, 3], [4, 5, 6]])
w.class # => Torch::Tensor
w.shape # => [2, 3]
w.ndim # => 2

w.numo # => [[1, 2, 3], [4, 5, 6]] (Numo::Int64#shape=[2,3])
```

## 1.1.2 矩阵的对于元素的运算

```ruby
w = Numo::NArray[[1, 2, 3], [4, 5, 6]]
x = Numo::NArray[[0, 1, 2], [3, 4, 5]]
w + x # => [[1, 3, 5], [7, 9 11]] (Numo::Int32#shape=[2,3])
w * x # => [[0, 2, 6], [12, 20, 30]] (Numo::Int32#shape=[2,3])
```

```ruby
w = Torch.tensor([[1, 2, 3], [4, 5, 6]])
x = Torch.tensor([[0, 1, 2], [3, 4, 5]])
w + x # => tensor([[1, 3, 5], [7, 9, 11]])
w * x # => tensor([[0, 2, 6], [12, 20, 30]])
```

## 1.1.3 不同形状的数组进行运算

```ruby
a = Numo::NArray[[1, 2], [3, 4]]
a * 10 # => [[10, 20], [30, 40]] (Numo::Int32#shape=[2,2])

a = Numo::NArray[[1, 2], [3, 4]]
b = Numo::NArray[[10, 20]]
a * b # => [[10, 40], [30, 80]] (Numo::Int32#shape=[2,2])
```

```ruby
a = Torch.tensor([[1, 2], [3, 4]])
a * 10 # => tensor([[10, 20], [30, 40]])

a = Torch.tensor([[1, 2], [3, 4]])
b = Torch.tensor([10, 20])
a * b # => tensor([[10, 40], [30, 80]])
```

## 1.1.4 向量内积和矩阵乘积

```ruby
a = Numo::NArray[1, 2, 3]
b = Numo::NArray[4, 5, 6]
a.dot(b) # => 32

a = Numo::NArray[[1, 2], [3, 4]]
b = Numo::NArray[[5, 6], [7, 8]]
a.dot(b) # => [[19, 22], [43, 50]] (Numo::Int32#shape=[2,2])
```

```ruby
a = Torch.tensor([1, 2, 3])
b = Torch.tensor([4, 5, 6])
a.dot(b) # => tensor(32) (Torch::Tensor)

a = Torch.tensor([[1, 2], [3, 4]])
b = Torch.tensor([[5, 6], [7, 8]])
a.matmul(b) # => tensor([[19, 22], [43, 50]])
```

## 1.2.1 神经网络推理

写出mini-batch版的全链接层变换

```ruby
w1 = Numo::DFloat.new(2, 4).rand
b1 = Numo::DFloat.new(4).rand
x = Numo::DFloat.new(10, 2).rand
h = x.dot(w1) + b1 # => (Numo::DFloat#shape=[10,4])
```

```ruby
w1 = Torch.rand(2, 4)
b1 = Torch.rand(4)
x = Torch.rand(10, 2)
h = x.matmul(w1) + b1 # => (Tensor#shape=[10,4])
```

增加sigmoid函数后的全链接层变换

```ruby
def sigmoid(number) = 1 / (1 + Math.exp(-number))

x = Numo::DFloat.new(10, 2).rand
w1 = Numo::DFloat.new(2, 4).rand
b1 = Numo::DFloat.new(4).rand
w2 = Numo::DFloat.new(4, 3).rand
b2 = Numo::DFloat.new(3).rand
h = x.dot(w1) + b1
a = h.map(&method(:sigmoid))
s = a.dot(w2) + b2
```

```ruby
x = Torch.rand(10, 2)
w1 = Torch.rand(2, 4)
b1 = Torch.rand(4)
w2 = Torch.rand(4, 3)
b2 = Torch.rand(3)
h = x.matmul(w1) + b1
a = h.sigmoid
s = a.matmul(w2) + b2 # => (Tensor#shape=[10,3])
```

## 1.2.2 层的类化及正向传播的实现

带sigmoid层的正向传播类

```ruby
# Using numo/narray
class Sigmoid
  attr_reader :params

  def initialize
    @params = []
  end

  def forward(x)
    x.map { |number| 1 / (1 + Math.exp(-number)) }
  end
end

class Affine
  attr_reader :params

  def initialize(weights, biases)
    @params = [weights, biases]
  end

  def forward(x)
    w, b = params
    x.dot(w) + b
  end
end

class TwoLayerNet
  attr_reader :layers, :params

  def initialize(input_size, hidden_size, output_size)
    # initialize weights and biases
    w1 = Numo::DFloat.new(input_size, hidden_size).rand
    b1 = Numo::DFloat.new(hidden_size).rand
    w2 = Numo::DFloat.new(hidden_size, output_size).rand
    b2 = Numo::DFloat.new(output_size).rand

    # generate layers
    @layers = [
      Affine.new(w1, b1),
      Sigmoid.new,
      Affine.new(w2, b2)
    ]

    # collect params
    @params = layers.inject([]) { |a, layer| a.concat(layer.params) }
  end

  def predict(x)
    layers.each { |layer| x = layer.forward(x) }

    x
  end
end
```

```ruby
# Using torch
class Sigmoid
  attr_reader :params

  def initialize
    @params = []
  end

  def forward(x)
    x.sigmoid
  end
end

class Affine
  attr_reader :params

  def initialize(weights, biases)
    @params = [weights, biases]
  end

  def forward(x)
    w, b = params
    x.matmul(w) + b
  end
end

class TwoLayerNet
  attr_reader :layers, :params

  def initialize(input_size, hidden_size, output_size)
    # initialize weights and biases
    w1 = Torch.rand(input_size, hidden_size)
    b1 = Torch.rand(hidden_size)
    w2 = Torch.rand(hidden_size, output_size)
    b2 = Torch.rand(output_size)

    # generate layers
    @layers = [
      Affine.new(w1, b1),
      Sigmoid.new,
      Affine.new(w2, b2)
    ]

    # collect params
    @params = layers.inject([]) { |a, layer| a.concat(layer.params) }
  end

  def predict(x)
    layers.each { |layer| x = layer.forward(x) }

    x
  end
end
```

使用TwoLayerNet类进行神经网络推理

```ruby
# Using numo-narray
x = Numo::DFloat.new(10, 2).rand
model = TwoLayerNet.new(2, 4, 3)
s = model.predict(x) # => (Numo::DFloat#shape=[10,3])
```

```ruby
# Using torch
x = Torch.rand(10, 2)
model = TwoLayerNet.new(2, 4, 3)
s = model.predict(x) # => (Tensor#shape[10,3])
```

## 1.3.4.4 计算图之sum节点的实现

```ruby
# Using numo/narray
d, n = 8, 7
x = Numo::DFloat.new(n, d).rand
y = x.sum(0).expand_dims(1)

dy = Numo::DFloat.new(1, d).rand
dx = dy.repeat(n, axis: 0) # => (Numo::DFloat#shape=[7,8])
```

```ruby
# Using torch
d, n = 8, 7
x = Torch.rand(n, d)
y = x.sum(0, true)

dy = Torch.rand(1, d)
dx = dy.repeat(n, 1) # => (Tensor#shape[7,8])
```

## 1.3.4.5 计算图之矩阵乘积（MatMul）节点实现为层类

```ruby
# Using numo/narray

class MatMul
  attr_reader :params, :grads, :x

  def initialize(weights = [])
    @params = [weights]
    @grads = [Numo::DFloat.zeros(weights.size)]
    @x = nil
  end

  def forward(x)
    @x = x
    weights = params[0]
    x.dot(weights)
  end

  def backward(dout)
    weights = params[0]
    dx = dout.dot(weights.transpose(0, 1))
    dw = x.transpose(0, 1).dot(dout)
    grads[0] = dw

    dx
  end
end
```

```ruby
# Using torch

class MatMul
  attr_reader :params, :grads, :x

  def initialize(weights = [])
    @params = [weights]
    @grads = [Torch.zeros[weights.size]]
    @x = nil
  end

  def forward(x)
    @x = x
    weights = params[0]
    x.matmul(weights)
  end

  def backward(dout)
    weights = params[0]
    dx = dout.matmul(weights.transpose(0, 1))
    dw = x.transpose(0, 1).matmul(dout)
    grads[0] = dw

    dx
  end
end
```

## 1.3.5 梯度的推导和反向传播实现

Sigmoid层

```ruby
# Using numo/narray

class Sigmoid
  attr_reader :params, :grads, :out

  def initialize
    @params = []
    @grads = []
    @out = nil
  end

  def forward(x)
    @out = 1 / (1 + Math.exp(-x))
  end

  def backward(dout)
    dx = dout * (1.0 - out) * out

    dx
  end
end
```

```ruby
# Using torch

class Sigmoid
  attr_reader :params, :grads, :out

  def initialize
    @params = []
    @grads = []
    @out = nil
  end

  def forward(x) # x(tensor)
    @out = x.sigmoid
  end

  def backward(dout)
    dout * (1.0 - out) * out
  end
end
```

Affine层

```ruby
# Using numo/narray
class Affine
  attr_reader :params, :grads, :x

  def initialize(weights, biases)
    @params = [weights, biases]
    @grads = [Numo::DFloat.zeros(weights.size), Numo::DFloat.zeros(biases.size)]
    @x = nil
  end

  def forward(x)
    @x = x
    w, b = params
    out = x.dot(w) + b
    out
  end

  def backward(dout)
    w, b = params
    dx = dout.dot(w.transpose) + b
    dw = x.transpose(0, 1).dot(dout)
    db = dout.sum(0)

    @grads[0] = dw
    @grads[1] = db

    dx
  end
end
```

```ruby
# Using torch
class Affine
  attr_reader :params, :grads, :x

  def initialize(weights, biases)
    @params = [weights, biases]
    @grads = [Torch.zeros(weights.size), Torch.zeros(biases.size)]
    @x = nil
  end

  def forward(x)
    @x = x
    w, b = params
    x.matmul(w) + b
  end

  def backward(dout)
    w, _b = params
    dx = dout.matmul(w.transpose(0, 1))
    dw = x.transpose(0, 1).matmul(dout)
    db = dout.sum(0)

    @grads[0] = dw
    @grads[1] = db

    dx
  end
end
```

Softmax 层

```ruby
class Softmax
  attr_reader :params, :grads, :out

  def initialize
    @params = []
    @grads= []
    @out = nil
  end

  def forward(x) # x(tensor with dtype:float)
    @out = Torch::NN::F.softmax(x)
  end

  def backward(dout)
    dx = out * dout
    sumdx = dx.sum(0)
    dx -= out * sumdx
    dx
  end
end
```

Softmax with Loss 层

```ruby
class SoftmaxWithLoss
  attr_reader :params, :grads
  attr_reader :y # softmax的输出
  attr_reader :t # t:监督标签

  def initialize
    @params = []
    @grads = []
  end

  def forward(x, t)
    @y = Torch::NN::F.softmax(x)

    # 在监督标签为one-hot向量的情况下，转换为正确解标签的索引
    @t = t.size == y.size ? t.argmax(1) : t

    Torch::NN::F.cross_entropy(@y, @t)
  end

  def backward(dout = 1)
    batch_size = t.shape[0]

    dx = y.clone
    dx[Torch.arange(batch_size), t] -= 1
    dx *= dout
    dx / batch_size
  end
end
```

## 1.3.6 权重的更新

梯度下降法(Stochastic gradient descent, SGD)的实现

```ruby
class SGD
  attr_reader :lr # learning rate 学习率

  def initialize(lr: 0.01)
    @lr = lr
  end

  def update(params, grads)
    params.map.with_index do |param, index|
      param.sub!(lr * grads[index])
    end
  end
end
```

## 1.4 使用神经网络解决问题

生成螺旋状数据集
```ruby
def load_data
  n = 100  # 各类的样本数
  dim = 2  # 数据的元素个数
  cls_num = 3  # 类别数

  x = Torch.zeros(n * cls_num, dim) # => (Torch::Tensor#shape[300,2])
  t = Torch.zeros(n * cls_num, cls_num, dtype: :int32)

  cls_num.times.with_index do |cls_index|
    n.times.with_index do |n_index|
      rate = n_index.to_f / n
      radius = 1.0 * rate
      theta = cls_index * 4.0 + rate * 4.0 + Torch.rand(1) * 0.2

      ix = n * cls_index + n_index
      x[ix] = [radius * theta.sin, radius * theta.cos].flatten
      t[ix, cls_index] = 1
    end
  end

  return x, t
end
```

图示螺旋状数据集
```ruby
require "matplotlib/pyplot"
plt = Matplotlib::Pyplot

x, t = load_data
puts "x: #{x.shape}"
puts "t: #{t.shape}"

n = 100
cls_num = 3
markers = %w[o x ^]

cls_num.times.map do |idx|
  plt.plot(x[idx*n...(idx+1)*n, 0].to_a, x[idx*n...(idx+1)*n, 1].to_a, marker: markers[idx])
end

plt.show()
```

神经网络的实现(two_layer_net.rb)

```ruby
class TwoLayerNet
  attr_reader :layers, :loss_layer, :params, :grads

  def initialize(input_size, hidden_size, output_size)
    w1 = 0.01 * Torch.rand(input_size, hidden_size)
    b1 = Torch.zeros(hidden_size)
    w2 = 0.01 * Torch.rand(hidden_size, output_size)
    b2 = Torch.zeros(output_size)

    @layers = [
      Affine.new(w1, b1),
      Sigmoid.new,
      Affine.new(w2, b2)
    ]

    @loss_layer = SoftmaxWithLoss.new

    @params = layers.inject([]) { |a, layer| a.concat(layer.params) }
    @grads = layers.inject([]) { |a, layer| a.concat(layer.grads) }
  end

  def predict(x)
    layers.each { |layer| x = layer.forward(x) }

    x
  end

  def forward(x, t)
    score = predict(x)
    loss_layer.forward(score, t)
  end

  def backward(dout = 1)
    dout = loss_layer.backward(dout)
    layers.reverse_each { |layer| dout = layer.backward(dout) }

    dout
  end
end
```

学习用的代码

```ruby
# 设定超参数
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = load_data
model = TwoLayerNet.new(2, hidden_size, 3)
optimizer = SGD.new(lr: learning_rate)

# 学习用的变量
data_size = x.length
max_iters = data_size / batch_size
total_loss = 0
loss_count = 0
loss_list = []

max_epoch.times.map do |epoch|
  # 打乱数据
  idx = Torch.randperm(data_size)
  x = x[idx]
  t = t[idx]

  max_iters.times.map do |iters|
    batch_x = x[iters*batch_size...(iters+1)*batch_size]
    batch_t = t[iters*batch_size...(iters+1)*batch_size]

    # 计算梯度，更新参数
    loss = model.forward(batch_x, batch_t)
    model.backward()
    optimizer.update(model.params, model.grads)

    total_loss += loss
    loss_count += 1

    # 定期输出学习过程
    if (iters+1) % 10 == 0
      avg_loss = total_loss / loss_count
      print("| epoch %d |  iter %d / %d | loss %.2f\n" % [epoch + 1, iters + 1, max_iters, avg_loss])
      loss_list.append(avg_loss)
      total_loss, loss_count = 0, 0
    end
  end
end
```

