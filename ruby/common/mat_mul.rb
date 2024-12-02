require "torch-rb"

class MatMul
  attr_reader :params, :grads, :x

  def initialize(weights = [])
    @params = weights
    @grads = Torch.zeros[weights.size]
    @x = nil
  end

  def forward(x)
    @x = x
    weights = params
    x.matmul(weights)
  end

  def backward(dout)
    weights = params
    dx = dout.matmul(weights.transpose(0, 1))
    dw = x.transpose(0, 1).matmul(dout)
    @grads = dw

    dx
  end
end
