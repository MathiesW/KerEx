from keras import saving, layers, Layer
from keras.src import ops
# from ...ops import large_negative_number, index_to_einsum_variable


# def _build_attention_equation(rank, attn_axes):
#     """
#     Builds einsum equations for the attention computation.

#     Query, key, value inputs after projection are expected to have the shape as:
#     `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
#     `bs` and `<non-attention dims>` are treated as `<batch dims>`.

#     The attention operations can be generalized:
#     1. Query-key dot product:
#         (<batch dims>, <query attention dims>, num_heads, channels),
#         (<batch dims>, <key attention dims>, num_heads, channels) ->
#         (<batch dims>, num_heads, <query attention dims>, <key attention dims>)
#     2. Combination:
#         (<batch dims>, num_heads, <query attention dims>, <key attention dims>),
#         (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch
#         dims>, <query attention dims>, num_heads, channels)

#     Parameters
#     ----------
#     rank: list | tuple
#         Rank of query, key, value tensors.
#     attn_axes: list | tuple
#         List/tuple of axes, `[-1, rank)`, that attention will be applied to.

#     Returns
#     -------
#     (dot_eq, combine_eq, attn_scores_rank) : tuple
#         Einsum equations for attention.

#     """

#     target_notation = ""
#     for i in range(rank):
#         target_notation += index_to_einsum_variable(i)
#     # `batch_dims` includes the head dim.
#     batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
#     letter_offset = rank
#     source_notation = ""
#     for i in range(rank):
#         if i in batch_dims or i == rank - 1:
#             source_notation += target_notation[i]
#         else:
#             source_notation += index_to_einsum_variable(letter_offset)
#             letter_offset += 1

#     product_notation = "".join(
#         [target_notation[i] for i in batch_dims]
#         + [target_notation[i] for i in attn_axes]
#         + [source_notation[i] for i in attn_axes]
#     )
#     dot_product_equation = "%s,%s->%s" % (
#         source_notation,
#         target_notation,
#         product_notation,
#     )
#     attn_scores_rank = len(product_notation)
#     combine_equation = "%s,%s->%s" % (
#         product_notation,
#         source_notation,
#         target_notation,
#     )
#     return dot_product_equation, combine_equation, attn_scores_rank


@saving.register_keras_serializable(package="KerasAddon.Layers.Attention", name="MultiHeadAttention")
class MultiHeadAttention(layers.MultiHeadAttention):
    """
    
    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    key_dim : int
        Size of each attention head for query and key.
    value_dim : int, optional
        Size of each attention head for value.
        If this is `None`, it is set to `key_dim`.
        Defaults to `None`.
    dropout : float, optional
        Dropout probability. Defaults to 0.
    use_bias : bool, optional
        If this is set, a bias is ued. Defaults to `True`.
    output_shape : tuple, optional
        The expected shape of an output tensor, besides the batch and sequence dims.
        If not specified, projects back to the query feature dim (the query input's last dimension).
        Defaults to `None`.
    attention_axes : int, optional
        Axes over which the attention is applied.
        `None` means attention over all axes, but batch, heads, and features.
        Defaults to `None`.
    kernel_initializer : str, optional
        Initializer for dense layer kernels. Defaults to `"glorot_uniform"`.
    bias_initializer : str, optional
        Initializer for dense layer biases. Defaults to `"zeros"`.
    kernel_regularizer : str, optional
        Regularizer for dense layer kernels. Defaults to `None`.
    bias_regularizer : str, optional
        Regularizer for dense layer biases. Defaults to `None`.
    activity_regularizer : str, optional
        Regularizer for dense layer activity. Defaults to `None`.
    kernel_constraint : str, optional
        Constraint for dense layer kernels. Defaults to `None`.
    bias_constraint : str, optional
        Constraint for dense layer kernels. Defaults to `None`.
    seed : int, optional
        Optional integer to seed the dropout layer. Defaults to `None`

    """

    def __init__(
            self, 
            num_heads, 
            key_dim, 
            value_dim=None, 
            dropout=0, 
            use_bias=True,  
            output_shape=None, 
            attention_axes=None, 
            kernel_initializer="glorot_uniform", 
            bias_initializer="zeros", 
            kernel_regularizer=None, 
            bias_regularizer=None, 
            activity_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None, 
            seed=None,
            **kwargs
        ):
        super().__init__(num_heads, key_dim, value_dim, dropout, use_bias, output_shape, attention_axes, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, seed, **kwargs)

    def _compute_attention_mask(
        self,
        query,
        value,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        use_causal_mask=False,
        prefix: int = 0,
        prediction_step: int = 0
    ):
        """
        Computes the attention mask, using the Keras masks of the inputs.

        * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
        * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
        * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
          mask is ignored if `key` is `None` or if `key is value`.
        * If `use_causal_mask=True`, then the causal mask is computed. Its shape
          is [1, T, S].

        All defined masks are merged using a logical AND operation (`&`).

        In general, if the `query` and `value` are masked, then there is no need
        to define the `attention_mask`.

        Parameters
        ----------
        query : KerasTensor
            Projected query tensor of shape `(B, T, N, key_dim)`.
        key : KerasTensor
            Projected key tensor of shape `(B, T, N, key_dim)`.
        value : KerasTensor
            Projected value tensor of shape `(B, T, N, value_dim)`.
        attention_mask : KerasTensor
            A boolean mask of shape `(B, T, S)`, that prevents attention to certain positions.
        use_causal_mask : bool
            A boolean to indicate whether to apply a causal mask to prevent tokens from attending to future tokens (e.g., used in a decoder Transformer).

        Returns
        -------
        attention_mask : KerasTensor
            A boolean mask of shape `(B, T, S)`, that prevents attention to certain positions, 
            based on the Keras masks of the `query`, `key`, `value`, and `attention_mask` tensors, 
            and the causal mask if `use_causal_mask=True`.

        """

        auto_mask = None
        if query_mask is not None:
            query_mask = ops.cast(query_mask, "bool")  # defensive casting
            # B = batch size, T = max query length
            auto_mask = ops.expand_dims(query_mask, -1)  # shape is [B, T, 1]
        if value_mask is not None:
            value_mask = ops.cast(value_mask, "bool")  # defensive casting
            # B = batch size, S == max value length
            mask = ops.expand_dims(value_mask, -2)  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if key_mask is not None:
            key_mask = ops.cast(key_mask, "bool")  # defensive casting
            # B == batch size, S == max key length == max value length
            mask = ops.expand_dims(key_mask, -2)  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if use_causal_mask:
            # the shape of the causal mask is [1, T, S]
            mask = self._compute_causal_mask(query, value, prefix, prediction_step)
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if auto_mask is not None:
            # merge attention_mask & automatic mask, to shape [B, T, S]
            attention_mask = (
                auto_mask
                if attention_mask is None
                else ops.cast(attention_mask, bool) & auto_mask
            )
        return attention_mask

    def _compute_causal_mask(self, query, value=None, prefix: int = 0, prediction_step: int = 0):
        """
        Computes a causal mask (e.g., for masked self-attention layers).

        For example, if query and value both contain sequences of length 4,
        this function returns a boolean tensor equal to:

        ```
        [[[True,  False, False, False],
          [True,  True,  False, False],
          [True,  True,  True,  False],
          [True,  True,  True,  True]]]
        ```

        Parameters
        ----------
        query : KerasTensor
            A query tensor of shape `(B, T, ...)`.
        value : KerasTensor
            A value tensor of shape `(B, S, ...)` (optional, defaults to query).

        Returns
        -------
        mask : KerasTensor
            A boolean tensor of shape `(1, T, S)` containing a lower triangular matrix of shape `(T, S)`.

        """

        q_seq_length = prediction_step or ops.shape(query)[1]
        v_seq_length = ops.shape(query)[1] if value is None else ops.shape(value)[1]
        ones_mask = ops.ones((1, q_seq_length, v_seq_length), dtype="int32")
        row_index = ops.cumsum(ones_mask, axis=-2)
        col_index = ops.cumsum(ones_mask, axis=-1)

        prefix = ops.full((1, q_seq_length, v_seq_length), prefix, dtype="int32")
        offset_mask = ops.greater_equal(prefix, col_index)
        offset_mask = ops.multiply(prefix, ops.cast(offset_mask, dtype="int32"))
        row_index = ops.add(row_index, offset_mask)

        mask = ops.greater_equal(row_index, col_index)
        return ops.pad(mask, pad_width=((0, 0), (0, v_seq_length - q_seq_length), (0, 0)))
    
    def call(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        return_attention_mask=False, 
        training=None,
        use_causal_mask=False,
        prefix=0,
        prediction_step=0
    ):
        if key is None:
            key = value

        attention_mask = self._compute_attention_mask(
            query,
            value,
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            prefix=prefix,
            prediction_step=prediction_step
        )

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask, training
        )
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        if return_attention_mask:
            return attention_output, attention_mask
        return attention_output


@saving.register_keras_serializable(package="KerasAddon.Layers.Attention", name="BaseAttention")
class BaseAttention(Layer):
    """
    Base multihead-attention layer as described in the paper "Attention is all you Need" [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).
    
    This BaseAttention class implements the Attention block, consisting in
    - Multihead-Attention layer, and
    - LayerNorm,
    and serves as a BaseLayer to inherit from.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    key_dim : int
        Size of each attention head for query and key.
    value_dim : int, optional
        Size of each attention head for value.
        If this is `None`, it is set to `key_dim`.
        Defaults to `None`.
    dropout : float, optional
        Dropout probability. Defaults to 0.
    use_bias : bool, optional
        If this is set, a bias is ued. Defaults to `True`.
    output_shape : tuple, optional
        The expected shape of an output tensor, besides the batch and sequence dims.
        If not specified, projects back to the query feature dim (the query input's last dimension).
        Defaults to `None`.
    attention_axes : int, optional
        Axes over which the attention is applied.
        `None` means attention over all axes, but batch, heads, and features.
        Defaults to `None`.
    kernel_initializer : str, optional
        Initializer for dense layer kernels. Defaults to `"glorot_uniform"`.
    bias_initializer : str, optional
        Initializer for dense layer biases. Defaults to `"zeros"`.
    kernel_regularizer : str, optional
        Regularizer for dense layer kernels. Defaults to `None`.
    bias_regularizer : str, optional
        Regularizer for dense layer biases. Defaults to `None`.
    activity_regularizer : str, optional
        Regularizer for dense layer activity. Defaults to `None`.
    kernel_constraint : str, optional
        Constraint for dense layer kernels. Defaults to `None`.
    bias_constraint : str, optional
        Constraint for dense layer kernels. Defaults to `None`.
    seed : int, optional
        Optional integer to seed the dropout layer. Defaults to `None`

    Notes
    -----
    With flash attention (Keras>3.6.0), the dropout probability has to be 0.

    """

    def __init__(
            self, 
            num_heads, 
            key_dim, 
            value_dim=None, 
            dropout=0, 
            use_bias=True,  
            output_shape=None, 
            attention_axes=None, 
            kernel_initializer="glorot_uniform", 
            bias_initializer="zeros", 
            kernel_regularizer=None, 
            bias_regularizer=None, 
            activity_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None, 
            seed=None,
            **kwargs
        ):
        super().__init__()

        self.mha: Layer = MultiHeadAttention(num_heads, key_dim, value_dim, dropout, use_bias, output_shape, attention_axes, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, seed, **kwargs)
        self.layernorm: Layer = layers.LayerNormalization()
        self.add: Layer = layers.Add()

    def build(self, query_shape, value_shape, key_shape=None):
        """
        Build method of BaseAttention

        Parameters
        ----------
        query_shape : tuple
            Input shape of query tensor.
        value_shape : tuple
            Input shape of value tensor.
        key_shape : tuple, optional
            Input shape of key tensor. Defaults to `None`.

        """

        if self.built:
            return
        
        self.mha.build(query_shape=query_shape, value_shape=value_shape, key_shape=key_shape)
        self.layernorm.build(input_shape=query_shape)
        self.add.build(input_shape=[query_shape, query_shape])

        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute output shape of layer

        Parameters
        ----------
        input_shape : tuple | list
            Shape of the input

        Returns
        -------
        output_shape : tuple
            Shape of the output (same as input!)

        """

        return tuple(input_shape)
    
    def call(self, *args):
        """
        Placeholder that ensures that the BaseAttention is not actually used.

        """

        raise NotImplementedError("Don't even think about calling this base layer!")


@saving.register_keras_serializable(package="KerasAddon.Layers.Attention", name="GlobalSelfAttention")
class GlobalSelfAttention(BaseAttention):
    """
    Global self-attention layer.
    
    This layer calculates the self-attention of the `query`.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    key_dim : int
        Size of each attention head for query and key.
    value_dim : int, optional
        Size of each attention head for value.
        If this is `None`, it is set to `key_dim`.
        Defaults to `None`.
    dropout : float, optional
        Dropout probability. Defaults to 0.
    use_bias : bool, optional
        If this is set, a bias is ued. Defaults to `True`.
    output_shape : tuple, optional
        The expected shape of an output tensor, besides the batch and sequence dims.
        If not specified, projects back to the query feature dim (the query input's last dimension).
        Defaults to `None`.
    attention_axes : int, optional
        Axes over which the attention is applied.
        `None` means attention over all axes, but batch, heads, and features.
        Defaults to `None`.
    kernel_initializer : str, optional
        Initializer for dense layer kernels. Defaults to `"glorot_uniform"`.
    bias_initializer : str, optional
        Initializer for dense layer biases. Defaults to `"zeros"`.
    kernel_regularizer : str, optional
        Regularizer for dense layer kernels. Defaults to `None`.
    bias_regularizer : str, optional
        Regularizer for dense layer biases. Defaults to `None`.
    activity_regularizer : str, optional
        Regularizer for dense layer activity. Defaults to `None`.
    kernel_constraint : str, optional
        Constraint for dense layer kernels. Defaults to `None`.
    bias_constraint : str, optional
        Constraint for dense layer kernels. Defaults to `None`.
    seed : int, optional
        Optional integer to seed the dropout layer. Defaults to `None`

    Notes
    -----
    With flash attention (Keras>3.6.0), the dropout probability has to be 0.

    """

    def call(self, x, return_attention_scores=False):
        """
        Call method of GlobalSelfAttention layer

        Parameters
        ----------
        x : KerasTensor
            Tensor for which the self-attention is calculated.
        return_attention_scores : bool, optional
            If `True`, the attention scores are returned. 
            Defaults to `False`.

        Returns
        -------
        y | (y, attention_scores) : KerasTensor | (KerasTensor, KerasTensor)
            The output of the self-attention.
            If `return_attention_scores==True`, the layer additionally returns the attention scores.

        """

        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            return_attention_scores=return_attention_scores
        )
        if return_attention_scores:
            attn_output, attn_scores = attn_output

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        if return_attention_scores:
            return x, attn_scores
        return x


@saving.register_keras_serializable(package="KerasAddon.Layers.Attention", name="CausalSelfAttention")
class CausalSelfAttention(BaseAttention):
    """
    Causal self-attention layer.
    
    This layer calculates the self-attention of the `query`.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    key_dim : int
        Size of each attention head for query and key.
    value_dim : int, optional
        Size of each attention head for value.
        If this is `None`, it is set to `key_dim`.
        Defaults to `None`.
    dropout : float, optional
        Dropout probability. Defaults to 0.
    use_bias : bool, optional
        If this is set, a bias is ued. Defaults to `True`.
    output_shape : tuple, optional
        The expected shape of an output tensor, besides the batch and sequence dims.
        If not specified, projects back to the query feature dim (the query input's last dimension).
        Defaults to `None`.
    attention_axes : int, optional
        Axes over which the attention is applied.
        `None` means attention over all axes, but batch, heads, and features.
        Defaults to `None`.
    kernel_initializer : str, optional
        Initializer for dense layer kernels. Defaults to `"glorot_uniform"`.
    bias_initializer : str, optional
        Initializer for dense layer biases. Defaults to `"zeros"`.
    kernel_regularizer : str, optional
        Regularizer for dense layer kernels. Defaults to `None`.
    bias_regularizer : str, optional
        Regularizer for dense layer biases. Defaults to `None`.
    activity_regularizer : str, optional
        Regularizer for dense layer activity. Defaults to `None`.
    kernel_constraint : str, optional
        Constraint for dense layer kernels. Defaults to `None`.
    bias_constraint : str, optional
        Constraint for dense layer kernels. Defaults to `None`.
    seed : int, optional
        Optional integer to seed the dropout layer. Defaults to `None`

    Notes
    -----
    With flash attention (Keras>3.6.0), the dropout probability has to be 0.

    """

    def call(self, x, prefix=0, prediction_step=0, return_attention_scores=False):
        """
        Call method of GlobalSelfAttention layer

        Parameters
        ----------
        x : KerasTensor
            Tensor for which the self-attention is calculated.
        prefix : int, optional
            Modifies the attention mask if `prefix>0`.
            Defaults to 0.
        prediction_step : int, optional
            Modifies the attention mask if `prediction_step>0`.
            Defaults to 0.
        return_attention_scores : bool, optional
            If `True`, the attention scores are returned. 
            Defaults to `False`.

        Returns
        -------
        y | (y, attention_scores) : KerasTensor | (KerasTensor, KerasTensor)
            The output of the self-attention.
            If `return_attention_scores==True`, the layer additionally returns the attention scores.

        """
        
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True,
            prefix=prefix,
            prediction_step=prediction_step,
            return_attention_scores=return_attention_scores
        )

        if return_attention_scores:
            attn_output, attn_scores = attn_output

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        if return_attention_scores:
            return x, attn_scores
        return x


@saving.register_keras_serializable(package="KerasAddon.Layers.Attention", name="CrossAttention")
class CrossAttention(BaseAttention):
    """
    Cross-attention layer.
    
    This layer calculates the cross-attention of the `query` and the `key`.
    The `key` represents the context that is merged with an input.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    key_dim : int
        Size of each attention head for query and key.
    value_dim : int, optional
        Size of each attention head for value.
        If this is `None`, it is set to `key_dim`.
        Defaults to `None`.
    dropout : float, optional
        Dropout probability. Defaults to 0.
    use_bias : bool, optional
        If this is set, a bias is ued. Defaults to `True`.
    output_shape : tuple, optional
        The expected shape of an output tensor, besides the batch and sequence dims.
        If not specified, projects back to the query feature dim (the query input's last dimension).
        Defaults to `None`.
    attention_axes : int, optional
        Axes over which the attention is applied.
        `None` means attention over all axes, but batch, heads, and features.
        Defaults to `None`.
    kernel_initializer : str, optional
        Initializer for dense layer kernels. Defaults to `"glorot_uniform"`.
    bias_initializer : str, optional
        Initializer for dense layer biases. Defaults to `"zeros"`.
    kernel_regularizer : str, optional
        Regularizer for dense layer kernels. Defaults to `None`.
    bias_regularizer : str, optional
        Regularizer for dense layer biases. Defaults to `None`.
    activity_regularizer : str, optional
        Regularizer for dense layer activity. Defaults to `None`.
    kernel_constraint : str, optional
        Constraint for dense layer kernels. Defaults to `None`.
    bias_constraint : str, optional
        Constraint for dense layer kernels. Defaults to `None`.
    seed : int, optional
        Optional integer to seed the dropout layer. Defaults to `None`

    Notes
    -----
    With flash attention (Keras>3.6.0), the dropout probability has to be 0.

    """

    def call(self, x, context, return_attention_scores=False):
        """
        Call method of GlobalSelfAttention layer

        Parameters
        ----------
        x : KerasTensor
            Input tensor (query).
        context : KerasTensor
            Context for `x` to calculate the attention.
        return_attention_scores : bool, optional
            If `True`, the attention scores are returned. 
            Defaults to `False`.

        Returns
        -------
        y | (y, attention_scores) : KerasTensor | (KerasTensor, KerasTensor)
            The output of the self-attention.
            If `return_attention_scores==True`, the layer additionally returns the attention scores.

        """
        attn_output = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=return_attention_scores
        )

        if return_attention_scores:
            attn_output, attn_scores = attn_output

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        if return_attention_scores:
            return x, attn_scores
        return x
