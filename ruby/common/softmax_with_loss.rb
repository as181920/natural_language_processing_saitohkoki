require "torch-rb"

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
