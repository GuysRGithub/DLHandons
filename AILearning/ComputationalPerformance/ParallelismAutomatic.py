from d2l import AllDeepLearning as d2l
from mxnet import nd, gpu, cpu


def run(x):
    return [nd.dot(x, x) for _ in range(10)]


x_cpu = nd.random.uniform(shape=(2000, 2000))
x_gpu = nd.random.uniform(shape=(2000, 2000), ctx=d2l.try_gpu())

run(x_cpu)
run(x_gpu)
nd.waitall()

# with d2l.benchmark("CPU&GPU time: %.4f sec"):
#     run(x_cpu)
#     run(x_gpu)
#     nd.waitall()


def copy_to_cpu(x):
    return [y.copyto(cpu()) for y in x]


# with d2l.benchmark('Run  on GPU: %.4f sec'):
#     y = run(x_gpu)
#     nd.waitall()
#
# with d2l.benchmark('Copy to CPU: %.4f sec'):
#     y_cpu = copy_to_cpu(y)
#     nd.waitall()

with d2l.benchmark('Run on GPU and copy to CPU: %.4f sec'):
    y = run(x_gpu)
    y_cpu = copy_to_cpu(y)
    nd.waitall()

