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
