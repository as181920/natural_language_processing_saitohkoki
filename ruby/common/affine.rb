require "torch-rb"

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
