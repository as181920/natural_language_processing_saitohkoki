require "torch-rb"
require_relative "../common/affine"
require_relative "../common/sigmoid"
require_relative "../common/softmax_with_loss"

class TwoLayerNet
  attr_reader :layers, :loss_layer

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
