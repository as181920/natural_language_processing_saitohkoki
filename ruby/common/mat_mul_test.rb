require_relative "../test/test_helper"
require_relative "../common/global"
require_relative "mat_mul"

describe MatMul do
  before do
    @c0 = Torch.tensor([[1, 0, 0, 0, 0, 0, 0]], dtype: Global::PRECISION)
    @c1 = Torch.tensor([[0, 0, 1, 0, 0, 0, 0]], dtype: Global::PRECISION)
    @w_in = Torch.randn(7, 3, dtype: Global::PRECISION)
    @w_out = Torch.randn(3, 7, dtype: Global::PRECISION)
    @in_layer0 = MatMul.new @w_in
    @in_layer1 = MatMul.new @w_in
    @out_layer = MatMul.new @w_out
  end

  it "forward for matmul layer" do
    # forward
    h0 = @in_layer0.forward(@c0)
    h1 = @in_layer1.forward(@c1)

    assert_equal [1, 3], h0.shape
    assert_equal [1, 3], h1.shape

    h = (h0 + h1) / 2
    s = @out_layer.forward(h)

    assert_equal [1, 7], s.shape
  end
end
