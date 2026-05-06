def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # 输入x的形状为[batch, sequence, features]
    with tf.variable_scope(scope):
        # 1. 线性映射：将输入 x 一次性映射为 Q (Query), K (Key), V (Value) 三个矩阵
        c = conv1d(x, "c_attn", n_state * 3)
        q, k, v = tf.split(c, 3, 2)

        # 2. 多头拆分：将 Q, K, V 拆分成多个注意力头（Multi-Head Attention）
        q = split_heads(q, hparams.n_head)
        k = split_heads(k, hparams.n_head)
        v = split_heads(v, hparams.n_head)

        # 3. 核心注意力计算（含 Mask 操作防止看到未来信息）
        # 这里会利用过去缓存的特征 (past) 提升推理速度，并返回当前特征 (present)
        a, present = multihead_attn(q, k, v, hparams=hparams)

        # 4. 合并多头：将多头注意力计算的结果重新拼接，并再做一次线性投影
        a = merge_heads(a)
        a = conv1d(a, "c_proj", n_state)

        return a, present


"""
【分析与作用总结】（写报告时用到）
1. 核心作用： 该函数实现了 GPT-2 的掩码多头自注意力机制（Masked Multi-Head Self-Attention）。正如幻灯片所说，它的作用是“将有限的注意力集中在重点信息上”，在生成每一个词时，决定应该分配多少注意力给前面的每一个词，从而理解上下文语境。
2. Q、K、V 机制： 函数首先通过 conv1d 将输入转化为 Q（查询，我需要寻找什么信息）、K（键，我包含什么特征）、V（值，我实际的语义内容）。
3. 单向注意力（Masking）： 因为 GPT-2 是解码器（Decoder-only）架构，只能根据上文预测下文。在内部的 multihead_attn 计算中，包含了一个遮罩（Mask）操作，把当前词汇之后的“未来信息”遮盖掉，强制模型只关注历史序列。
"""
