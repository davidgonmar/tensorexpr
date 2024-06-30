import numpy as np  # noqa


class Tensor:
    def __init__(self, *dims, op="load", srcs=[]):
        assert len(dims) == len(set(dims)), "dims must be unique"
        self.dims = list(dims)
        self.op = op
        self.srcs = srcs

    def __getitem__(self, key):
        assert len(key) == len(
            self.dims
        ), "key must have the same length as the number of dimensions"
        return self

    def __setitem__(self, key, value):
        assert len(key) == len(
            self.dims
        ), "key must have the same length as the number of dimensions"
        self.op = value.op
        self.srcs = value.srcs

    def compile(
        self, ctx={"src": "", "names": {}, "namecounts": {}, "args": []}, root=True
    ):
        for src in self.srcs:
            src.compile(ctx, False)

        if self.op == "mul":
            # get the source tensors and their dimensions
            src1, src2 = self.srcs
            # get the dimensions to contract
            contract_dims = set(src1.dims).intersection(set(src2.dims))
            
            # now render the code with a tensordot
            src1_name = ctx["names"][src1]
            src2_name = ctx["names"][src2]

            src = f"np.tensordot({src1_name}, {src2_name}, axes=({', '.join([str(src1.dimidx(dim)) for dim in contract_dims])}, {', '.join([str(src2.dimidx(dim)) for dim in contract_dims])}))"

            # store the source code
            ctx["src"] += src
            if not root:
                return

        if self.op == "load":
            namec = ctx["namecounts"].get(self.op, 0)
            name = "load" + str(namec)
            ctx["namecounts"][self.op] = namec + 1
            ctx["names"][self] = name
            ctx["args"].append(name)
            if not root:
                return

        # eval the source code
        src = ctx["src"]
        src = f"def fn({', '.join(ctx['args'])}):\n    return {src}"
        print(src)

        # return a function to evaluate the tensor
        def retfn(*args):
            # create a dictionary of the arguments
            argdict = {}
            for i, arg in enumerate(args):
                argdict[self.dims[i].name] = arg

            # execute the source code
            exec(src, globals(), argdict)
            return argdict["fn"](*args)

        return retfn

    def __repr__(self):
        return f"Tensor({', '.join([dim.name for dim in self.dims])}, op='{self.op}')"

    def __mul__(self, other):
        # union of dimensions
        new_dims = []
        for dim in self.dims:
            new_dims.append(dim)
        for dim in other.dims:
            if dim not in new_dims:
                new_dims.append(dim)
        return Tensor(*new_dims, op="mul", srcs=[self, other])

    def dimidx(self, dim):
        return self.dims.index(dim)


class Dim:
    def __hash__(self):
        return hash(self.name)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return "Dim('{}')".format(self.name)


if __name__ == "__main__":
    i = Dim("i")
    j = Dim("j")
    k = Dim("k")

    t = Tensor(i, j)  # Tensor with 3 dimensions
    t2 = Tensor(j, k)  # Tensor with 2 dimensions

    t3 = Tensor(i, k)

    t3[i, k] = t[i, j] * t2[j, k]

    fn = t3.compile()

    print(fn)

    a = np.random.rand(3, 4)
    b = np.random.rand(4, 5)

    print(fn(a, b))

    print(np.dot(a, b))  # should be the same
