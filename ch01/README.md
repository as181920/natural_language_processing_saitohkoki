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
h = x.matmul(w1) + b1 # => (tensor#shape=[10,4])
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
s = a.matmul(w2) + b2 # => (tensor#shape=[10,3])
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
    @params = layers.map(&:params)
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
    @params = layers.map(&:params)
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
s = model.predict(x) # => (tensor#shape[10,3])
```
