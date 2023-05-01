from base import module

class RWKVTFOps(module):
    def __init__(self, layers, embed, *args, useGPU: bool = True, **kwargs):
        
        import tensorflow as tf
        
        if (not useGPU):
            tf.config.experimental.set_visible_devices([], "GPU")
        tf.config.optimizer.set_jit(True)
        tf.config.optimizer.set_experimental_options(
            {"auto_mixed_precision": True})

        super(RWKVTFOps, self).__init__(layers, embed, *args, **kwargs)
        self.initTensor = lambda x: tf.convert_to_tensor(
            x.float().cpu().numpy())
        self.mnstack = tf.stack

        self.sqrt = tf.sqrt
        self.mean = tf.reduce_mean
        self.relu = lambda x: tf.maximum(x, tf.zeros_like(x))
        self.minimum = tf.minimum
        self.maximum = tf.maximum
        self.exp = tf.exp
        self.unsqueeze = tf.expand_dims

        def intTensor(x): return tf.convert_to_tensor(
            x if type(x) == list else [x], dtype=tf.int64) if type(x) != tf.Tensor else x
        self.intTensor = intTensor
        self.cat = lambda x: tf.concat(x, axis=0)

        def matvec(x, y):
            y = tf.transpose(y)
            return tf.transpose(tf.matmul(x, y))
        self.matvec = matvec
        self.prod = lambda x: tf.reduce_prod(x, axis=1)

        def roll(x):
            zx = x
            rx = tf.concat([zx[:1], zx[0:-1]], axis=0)
            return rx
        self.rng = lambda x: tf.range(x, dtype=tf.int32)

        def emptyarray(x): return tf.TensorArray(tf.float32, size=x, element_shape=(
            tf.TensorShape([embed])
        ))
        self.emptyarray = emptyarray
        def arrayPush(x, y, i): return x.write(i, y)
        self.arrayPush = arrayPush
        def arrayGet(x, i): return x.read(i)
        self.arrayGet = arrayGet

        def stack(x):
            return tf.stack(x) if type(x) == list or type(x) == tuple else x.stack()
        self.stack = stack
        self.roll = roll
        def pop(x): return x[-1]
        self.pop = pop

        def push(x, y):
            # x[0] = y
            # tensorflow does not support item assignment
            # so we have to do this
            return tf.concat([tf.expand_dims(y, 0), x[1:]], axis=0)

        self.push = push
        self.log = tf.math.log
        self.lerp = lambda x, y, z: x*(1-z)+y*z
       # module def
        self.module = tf.Module

        self.divide = tf.math.truediv
        # class def
       # tensorflow function defs
        self.initfunc = lambda x: x
        self.layerdef = tf.function(
            input_signature=[tf.TensorSpec(shape=[None, embed], dtype=tf.float32), tf.TensorSpec(shape=[None, None], dtype=tf.float32), tf.TensorSpec(dtype=tf.int32, shape=None)])

        self.mainfunc = tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(
            shape=[self.n_layers, (4+self.useLogFix), embed], dtype=tf.float32)])
        self.emptyState = tf.convert_to_tensor(
            self.emptyState, dtype=tf.float32)

        # self.scatterindices = [slice(2, 5), slice(2, 4), slice(0, 2)]

        # def scatter(x, y, z):
        #     x = x.at[y].set(z)
        #     return x
        # work around for tensorflow not supporting item assignment
        def scatter(x, y, z):
            rx = tf.concat([x[:y[0]], z, x[y[1]:]], axis=0) if type(
                y) == tuple else tf.reshape(tf.concat([x[:y], [z], x[y+1:]], axis=0), ((layers, 4+self.useLogFix, embed)))

            return rx

        self.scatter = scatter

        self.scatterindices = [(2, 5), (2, 4), (0, 2)]

        def ln(x, w, b):
            # layernorm batchnorm
            xee2 = x - tf.reduce_mean(x, 1, True)

            x2 = tf.sqrt(tf.reduce_mean(xee2*xee2, 1, True) +
                         1e-5)
            o = w*(xee2/x2) + b
            return o

        self.layernorm = ln

        self.len = lambda x: tf.shape(x)[0]
        self.getIndex = lambda x, y: tf.gather(x, y, axis=0)
