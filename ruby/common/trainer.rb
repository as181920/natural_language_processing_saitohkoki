require "active_support/all"
require "debug"
require "matplotlib/pyplot"
require "torch-rb"

class Trainer
  attr_reader :model, :optimizer, :loss_list, :eval_interval, :current_epoch

  def initialize(model:, optimizer:)
    @model = model
    @optimizer = optimizer
    @loss_list = []
    @eval_interval = nil
    @current_epoch = 0
  end

  def fit(x, t, max_epoch: 10, batch_size: 32, _max_grad: nil, eval_interval: 20) # rubocop:disable Metrics/MethodLength, Metrics/ParameterLists, Metrics/AbcSize
    @eval_interval = eval_interval
    data_size = x.length
    max_iters = data_size / batch_size
    total_loss = 0
    loss_count = 0

    Time.now.to_f

    max_epoch.times.map do |_epoch|
      # 打乱数据
      idx = Torch.randperm(data_size)
      x = x[idx]
      t = t[idx]

      max_iters.times.map do |iters|
        batch_x = x[iters * batch_size...(iters + 1) * batch_size]
        batch_t = t[iters * batch_size...(iters + 1) * batch_size]

        # 计算梯度，更新参数
        loss = model.forward(batch_x, batch_t)
        model.backward
        # params, grads = remove_duplicate(model.params, model.grads)
        # clip_grads(grads, max_grad) if max_grad.present?
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 评价
        next unless eval_interval.present? && (iters % eval_interval).zero?

        avg_loss = total_loss / loss_count
        print(format("| epoch %d |  iter %d / %d | loss %.2f\n", current_epoch + 1, iters + 1, max_iters, avg_loss)) # rubocop:disable Style/FormatStringToken
        @loss_list.append(avg_loss)
        total_loss = 0
        loss_count = 0
      end

      @current_epoch += 1
    end
  end

  def plot(ylim: nil)
    x = Torch.arange(loss_list.length)
    plt = Matplotlib::Pyplot
    plt.ylim(*ylim) if ylim.present?

    plt.plot(x.to_a, loss_list.map(&:to_f), label: "train")
    plt.xlabel("iterations (x#{eval_interval})")
    plt.ylabel("loss")
    plt.show
  end
end
