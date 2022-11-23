import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

batch_size = 2
# 更改数据类型torch.dtype或者设备torch.device
# 两个源句子长度为2和4
src_len = torch.Tensor([2, 4]).to(torch.int32)
# 两个目标句子，长度为4和3
tgt_len = torch.Tensor([4, 3]).to(torch.int32)
# print(src_len)
# print(tgt_len)

# 单词表的大小
max_num_src_words = 8
max_num_tgt_words = 8

# 序列的最大长度
max_src_seq_len = 5
max_tgt_seq_len = 5
model_dim = 8

# 构建出两个句子，这里存储的是句子的索引
# src_seq = [torch.randint(1, 8, (L,)) for L in src_len]
#  F.pad (左边填充数， 右边填充数， 上边填充数， 下边填充数)
src_seq = [F.pad(torch.randint(1, max_num_src_words, (L,)), (0, max(src_len) - L)) for L in src_len]
tgt_seq = [F.pad(torch.randint(1, max_num_tgt_words, (L,)), (0, max(src_len) - L)) for L in tgt_len]
# 把样本拼接在一起
"""
函数目的： 在给定维度上对输入的张量序列seq 进行连接操作。
outputs = torch.cat(inputs, dim=?) → Tensor
参数
inputs : 待连接的张量序列，可以是任意相同Tensor类型的python 序列
dim : 选择的扩维, 必须在0到len(inputs[0])之间，沿着此维连接张量序列。

"""
src_seq = torch.cat([torch.unsqueeze(i, 0) for i in src_seq], dim=0)
tgt_seq = torch.cat([torch.unsqueeze(i, 0) for i in tgt_seq], dim=0)
# print("src_seq ", src_seq)
# print("tgt_seq ", tgt_seq)

#  ###############################################################
#  构造embedding
#  ###############################################################
src_embedding_table = nn.Embedding(max_num_src_words + 1, model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words + 1, model_dim)

# print("src_embedding_table", src_embedding_table.weight)

src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)
# print("src_embedding", src_embedding)

#  ###############################################################
# 构建position embedding
#  ###############################################################
max_position_len = 5
pos_mat = torch.arange(max_position_len).reshape((-1, 1))
i_mat = torch.pow(10000, torch.arange(0, 8, 2).reshape((1, -1)) / model_dim)
# print("pos_mat", pos_mat)
# print("i_mat", i_mat)
"""
import numpy as np
X = np.random.rand(6)
print(X)
[0.13678488 0.49035755 0.05431084 0.86536952 0.66651492 0.88161923]

print(X[0:6]) # 输出Index从0 Index为5的所有数 数从0开始计数 6为最后一个数+1
[0.13678488 0.49035755 0.05431084 0.86536952 0.66651492 0.88161923]

print(X[::3]) # 跳跃读取 读取的数的Index之间相差3
[0.13678488 0.86536952]

print(X[1::2]) # 从Index为1的位置开始跳跃读取 读取的Index之间相差为2
[0.49035755 0.86536952 0.88161923]

print(X[::-1]) # 翻转读取
[0.88161923 0.66651492 0.86536952 0.05431084 0.49035755 0.13678488]

print(X[1::-1]) # 从后向前读取 Index为1
[0.49035755 0.13678488]
"""
pe_embedding_table = torch.zeros(max_position_len, model_dim)
pe_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat)
pe_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat)
# print(pe_embedding_table)

pe_embedding = nn.Embedding(max_position_len, model_dim)
pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False)
# print(pe_embedding.weight.shape)

"""
torch.stack
沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
浅显说法：把多个2维的张量凑成一个3维的张量；
多个3维的凑成一个4维的张量…以此类推，也就是在增加新的维度进行堆叠。
outputs = torch.stack(inputs, dim=?) → Tensor
"""
src_pos = torch.stack([torch.arange(max(src_len)) for _ in src_len])
tgt_pos = torch.stack([torch.arange(max(tgt_len)) for _ in tgt_len])
# print(src_pos)
src_pe_embedding = pe_embedding(src_pos)
tgt_pe_embedding = pe_embedding(tgt_pos)
# print(tgt_pe_embedding)

#  ###############################################################
# 构造encoder的self-attention mask
#  ###############################################################
# mask的shape: [batch_size, max_src_len, max_src_len]
valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len) - L), value=0), 0) \
                                               for L in src_len]), 2)
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1, 2))
invalid_encoder_pos_matrix = 1 - valid_encoder_pos_matrix
mask_encoder_self_attention = invalid_encoder_pos_matrix.to(torch.bool)
# print("mask_encoder_self_attention ", mask_encoder_self_attention)

score = torch.randn(batch_size, max(src_len), max(src_len))

"""
masked_fill_(mask, value)
掩码操作
用value填充tensor中与mask中值为1位置相对应的元素。mask的形状必须与要填充的tensor形状一致。
"""
masked_score = score.masked_fill(mask_encoder_self_attention, -np.inf)
prob = F.softmax(masked_score, -1)
# print(src_len)
# print(score)
# print(masked_score)
# print(prob)


# step5 构造intra-attention的mask
# Q @ K^T shape:[batch_size, tgt_seq_len, src_seq_len]
valid_encoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len) - L), value=0), 0) \
                                               for L in src_len]), 2)
valid_decoder_pos = torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(tgt_len) - L), value=0), 0) \
                                               for L in tgt_len]), 2)
# print(valid_encoder_pos)
# print(valid_decoder_pos)
"""
torch.mm是两个矩阵相乘，即两个二维的张量相乘
torch.bmm(input, mat2, *, deterministic=False, out=None) → Tensor
对 input 和 mat2 矩阵执行批处理矩阵积。
input 和 mat2 必须是三维张量，每个张量包含相同数量的矩阵。
"""
valid_cross_pos_matrix = torch.bmm(valid_decoder_pos, valid_encoder_pos.transpose(1, 2))
invalid_cross_pos_matrix = 1 - valid_cross_pos_matrix
mask_cross_attention = invalid_cross_pos_matrix.to(torch.bool)
# print(mask_cross_attention)


# step6 构造decoder self-attention mask
"""
torch.tril下三角矩阵
"""
valid_decoder_tri_matrix = torch.cat([
    torch.unsqueeze(
        F.pad(torch.tril(torch.ones((L, L))), (0, max(tgt_len) - L, 0, max(tgt_len) - L), value=0), 0)
    for L in tgt_len])
# print(valid_decoder_tri_matrix)
# valid_cross_tri_matrix = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1, 2))
# print(valid_cross_tri_matrix)
invalid_decoder_tri_matrix = 1 - valid_decoder_tri_matrix
invalid_decoder_tri_matrix = invalid_decoder_tri_matrix.to(torch.bool)
# print(invalid_decoder_tri_matrix)
"""
masked_fill_(mask, value)
掩码操作
用value填充tensor中与mask中值为1位置相对应的元素。mask的形状必须与要填充的tensor形状一致。
"""
score = torch.randn(batch_size, max(tgt_len), max(tgt_len))
masked_score = score.masked_fill(invalid_decoder_tri_matrix, -1e9)
prob = F.softmax(masked_score, -1)
print(prob)
# step7 构建scaled self-attention
def scaled_dot_product_attention(Q, K, V, attn_mask, model_dim):
    # shape of Q K V : [batch_size*num_head, seq_len, model_dim/num_head
    score = torch.bmm(Q, K.transpose(-2, -1)) / torch.sqrt(model_dim)
    masked_score = score.masked_fill(attn_mask, -1e9)
    prob = F.softmax(masked_score, -1)
    context = torch.bmm(prob, V)
    return context




