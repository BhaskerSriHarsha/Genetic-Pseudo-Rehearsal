def test(SHAPE):
    print(type(SHAPE))
    print(SHAPE)

def AddTuple(TUPLE, ELEMENT):
    TUPLE = list(TUPLE)
    TUPLE.insert(0, ELEMENT)

    return tuple(TUPLE)

shape = (784,)
shape = AddTuple(shape, 1)

test(shape)
