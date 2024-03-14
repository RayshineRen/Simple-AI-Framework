import simpleflow as sf

with sf.Graph().as_default():
    a = sf.Constant([1.0, 2.0], name="a")
    b = sf.Constant(2.0, name="b")
    c = sf.add(a, b, name="c")
