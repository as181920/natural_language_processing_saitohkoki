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
