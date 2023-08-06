require "debug"
require "matplotlib/pyplot"
require "torch-rb"

require_relative "../common/affine"
require_relative "../common/sgd"
require_relative "../common/sigmoid"
require_relative "../common/softmax_with_loss"
require_relative "../dataset/spiral"
require_relative "./two_layer_net"

# 学习用的代码

## 设定超参数
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = Dataset::Spiral.load_data
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
x, t = Dataset::Spiral.load_data
n = 100
cls_num = 3
markers = %w[o x ^]
cls_num.times.map do |idx|
  plt.plot(x[idx*n...(idx+1)*n, 0].to_a, x[idx*n...(idx+1)*n, 1].to_a, marker: markers[idx])
end
plt.show
