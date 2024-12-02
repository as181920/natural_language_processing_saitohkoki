require "debug"
require "matplotlib/pyplot"
require "torch-rb"

require_relative "../common/affine"
require_relative "../common/sgd"
require_relative "../common/sigmoid"
require_relative "../common/softmax_with_loss"
require_relative "../common/trainer"
require_relative "../dataset/spiral"
require_relative "two_layer_net"

# 设定超参数
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = Dataset::Spiral.load_data
model = TwoLayerNet.new(2, hidden_size, 3)
optimizer = SGD.new(lr: learning_rate)

trainer = Trainer.new(model:, optimizer:)
trainer.fit(x, t, max_epoch:, batch_size:, eval_interval: 10)
trainer.plot
