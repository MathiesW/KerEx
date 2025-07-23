class _BaseConvBlock:
    """
    This class just provides basic functionality for convolutional blocks

    **Use for inheritance only!**

    """

    def set_vars(self, filters, **kwargs):
        """
        Wrap all `**kwargs` to list and set as class attribute

        The `filters` parameter has a special role here, as it determines the depth of the network.
        All other parameters (if given as a list or tuple initially) have to comply with the length of `filters`.

        Parameters
        ----------
        filters : int | list | tuple
            Number of filters of the `Encoder` block.
        **kwargs : Additional optional keyword arguments, may include `kernel_size`, `strides`, `dilation_rate`, and `groups`

        Raises
        ------
        ValueError
            If an argument is a list/tuple and contains other dtypes but int or tuple.
        ValueError
            If an argument is a list/tuple and contains tuples that do not match `self.rank`.
        ValueError
            If an argument does not match the length of `filters`.
        
        """

        # add reference `filters` at first position in kwargs
        kwargs = {"filters": filters, **kwargs}

        # now iterate over kwargs
        for k, v in kwargs.items():
            if isinstance(v, (list, tuple)):
                if len(v) == 1:
                    # unpack list/tuple of length 1
                    v, = v

            if isinstance(v, (int, str, bool, tuple)):
                if hasattr(self, "filters"):
                    # wrap singular argument (that is not `filters` in list)
                    v = [v] * len(self.filters)
                else:
                    # wrap `filters` argument in list
                    v = [v]

            if not all(isinstance(f, (int, str, tuple)) for f in v):
                raise ValueError(f"Received bad `{k}` argument. Expected all entries of `{k}` to be either `int`, `str`, or `tuple`.")
            
            if not all(len(f) == self.rank for f in v if isinstance(f, tuple)):
                raise ValueError(f"Rank of provided filters does not match rank of the model, expected `{k}` of rank {self.rank}.")
            
            try:
                if not len(v) == len(self.filters):
                    raise ValueError(f"Too many arguments for `{k}`, expected {len(self.filters)}, received {len(v)}.")
            except AttributeError:
                # `self.filters` was not set yet, no comparison possible
                pass
            
            # set `v` as class attribute `k`
            setattr(self, k, v)