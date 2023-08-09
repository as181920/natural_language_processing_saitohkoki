require "numo/narray"
require "matplotlib/pyplot"

a = Numo::DFloat.new(3, 5).seq
p a

plt = Matplotlib::Pyplot

xs = [*1..100].map {|x| (x - 50) * Math::PI / 100.0 }
ys = xs.map {|x| Math.sin(x) }

plt.plot(xs, ys)
plt.show()
