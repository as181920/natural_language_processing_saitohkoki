require "debug"
require "matplotlib/pyplot"
require "torch-rb"

class Sigmoid
  attr_reader :params, :grads, :out

  def initialize
    @params = []
    @grads = []
    @out = nil
  end

  # x: (Tensor)
  def forward(x)
    @out = x.sigmoid
  end

  def backward(dout)
    dout * (1.0 - out) * out
  end
end

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

class SoftmaxWithLoss
  attr_reader :params, :grads, :y, :t # y:softmax的输出, t:监督标签

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

class TwoLayerNet
  attr_reader :layers, :loss_layer

  def initialize(input_size, hidden_size, output_size) # rubocop:disable Metrics/MethodLength
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
  end

  def params
    layers.inject([]) { |a, layer| a.concat(layer.params) }
  end

  def grads
    layers.inject([]) { |a, layer| a.concat(layer.grads) }
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

## 生成螺旋状数据集

def load_data # rubocop:disable Metrics/MethodLength
  n = 100  # 各类的样本数
  dim = 2  # 数据的元素个数
  cls_num = 3 # 类别数

  x = Torch.zeros(n * cls_num, dim) # => (Torch::Tensor#shape[300,2])
  t = Torch.zeros(n * cls_num, cls_num, dtype: :int32)

  cls_num.times do |cls_index|
    n.times do |n_index|
      rate = n_index.to_f / n
      radius = 1.0 * rate
      theta = (cls_index * 4.0) + (rate * 4.0) + (Torch.rand(1) * 0.2)

      ix = (n * cls_index) + n_index
      x[ix] = [radius * theta.sin, radius * theta.cos].flatten
      t[ix, cls_index] = 1
    end
  end

  [x, t]
end

# 学习用的代码

## 设定超参数

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
    model.backward
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

# 绘制学习结果
plt = Matplotlib::Pyplot
plt.plot(Torch.arange(loss_list.length).to_a, loss_list.map(&:to_f), label: "train")
plt.xlabel("iterations (x10)")
plt.ylabel("loss")
plt.show

# 绘制决策边界
h = 0.001
x_min, x_max = x[0..-1, 0].min - 0.1, x[0..-1, 0].max + 0.1
y_min, y_max = x[0..-1, 1].min - 0.1, x[0..-1, 1].max + 0.1
xx, yy = Torch.meshgrid([Torch.arange(x_min, x_max, h), Torch.arange(y_min, y_max, h)], indexing: "xy")
score = model.predict(Torch.column_stack([xx.ravel, yy.ravel]))
predict_cls = score.argmax(1)
plt.contourf(xx.to_a, yy.to_a, predict_cls.reshape(xx.shape).to_a) # OPTIMIZE: can not plot without .to_a, but this conversation consume much time
plt.axis("off")

# 绘制数据点
x, t = load_data
n = 100
cls_num = 3
markers = %w[o x ^]
cls_num.times.map do |idx|
  plt.plot(x[idx*n...(idx+1)*n, 0].to_a, x[idx*n...(idx+1)*n, 1].to_a, marker: markers[idx])
end
plt.show
