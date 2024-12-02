module Dataset
  module Spiral
    module_function

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
  end
end
