# FELLRec 系统说明

## GPU资源需求

### BigRec_FELLRec（基于LLaMA的方案）
根据 `train.sh` 配置：

- **GPU数量**: 4块GPU（CUDA_VISIBLE_DEVICES=0,1,2,3）
- **GPU显存要求**: 
  - 使用 `load_in_8bit=True` 进行8-bit量化，显著降低显存需求
  - 使用 LoRA 进行参数高效微调（lora_r=8, lora_alpha=16）
  - 每块GPU需要 **12GB 显存**（LLaMA-7B模型）
- **训练配置**:
  - 分布式训练（DistributedDataParallel）
  - 批次大小：64（micro_batch_size=4，gradient_accumulation_steps=16）
  - 多进程数：4

### RecFormer_FELLRec（基于RecFormer的方案）
- GPU数量取决于 `accelerate.yaml` 中的配置
- 使用LongFormer作为backbone
- 相对较小的显存需求（8GB/GPU）

## 系统架构：如何将推荐系统、LLM与联邦学习结合

### 1. 推荐系统数据转换

推荐任务被转换为**生成式任务**，使用类似Alpaca的指令微调格式：

```
### Instruction:
Given a list of video games the user has played before, please recommend a new video game that the user likes to the user.

### Input:
The user has played the following video games before:
"SOCOM U.S. Navy Seals", "Deal or No Deal", ...

### Response:
"Diablo III"
```

数据示例：
- `instruction`: 推荐任务描述
- `input`: 用户历史交互物品列表
- `output`: 推荐的物品
- `user`: 用户ID

### 2. 联邦学习架构

#### 2.1 客户端划分（`split_dataset`函数）

使用**KMeans聚类**基于用户embedding将用户分到不同客户端：

```python
# 1. 加载预训练的Matrix Factorization模型
MF_model = torch.load(pretrain_emb_path)

# 2. 提取所有用户的embedding
user_embeddings = MF_model['embedding_user.weight']

# 3. K-means聚类（默认5个客户端）
kmeans = KMeans(n_clusters=client_num).fit(user_embeddings)

# 4. 根据聚类结果分配用户到对应客户端
```

相似兴趣的用户被分配到同一个客户端，实现数据分布的异质性模拟。

#### 2.2 训练流程

**每个epoch的步骤**：

1. **客户端本地训练**：
   ```python
   for i in range(client_num):
       # 每个客户端独立训练自己的模型
       client_trainer[i].train()
       # 保存客户端模型
       client[i].save_pretrained(f'{output_dir}/client{i}_{save_name}')
   ```

2. **聚合阶段**（`aggregate`函数）：
   ```python
   # 计算客户端模型参数的余弦相似度
   sim_matrix = cluster_clients(accumulated_params)
   
   # 基于相似度的加权聚合
   for i in range(client_num):
       warm_weight[i] = math.tanh(alpha/(train_loss[i]**(epoch+1/beta)))
       # 根据相似度和损失动态聚合
       lora_weight = get_aggregate_lora_weight(i, sim_matrix, accumulated_params, warm_weight[i], beta)
   ```

3. **模型更新**：
   - 使用余弦相似度矩阵（`sim_matrix`）计算客户端间的相似性
   - 基于训练损失动态调整聚合权重（`warm_weight`）
   - 实现选择性聚合，相似客户端分享更多知识

### 3. LoRA参数高效微调

使用**PEFT（Parameter-Efficient Fine-Tuning）**：

```python
config = LoraConfig(
    r=8,                    # LoRA rank
    lora_alpha=16,          # 缩放参数
    target_modules=["q_proj", "v_proj"],  # 只微调注意力层
    lora_dropout=0.05,
)

model = get_peft_model(base_model, config)
```

**优势**:
- 只训练少量参数（~1-2%的原模型参数）
- 降低显存需求
- 联邦学习中传输的参数更少

### 4. Split Learning（可选）

在 `utils.py` 中的 `split_client_server` 函数支持：

```python
def split_client_server(original_model, k):
    # 前k层保留在客户端
    client_layers = model.layers[:k] + [model.layers[-1]]
    
    # 中间层在服务器
    server_layers = model.layers[k:-1]
```

实现**分层联邦学习**，提供额外的隐私保护。

### 5. 核心创新点

1. **基于余弦相似度的客户端聚类**：
   ```python
   similarity_matrix = cosine_similarity(params_matrix)
   ```
   相似客户端的模型参数具有更高的聚合权重。

2. **动态权重调整**：
   ```python
   warm_weight[i] = math.tanh(alpha/(train_loss[i]**(epoch+1/beta)))
   ```
   训练损失低的客户端在聚合中获得更高权重。

3. **联邦学习 + LoRA**：
   - 聚合时只传输LoRA权重
   - 降低通信成本
   - 提高隐私保护

### 6. 两种实现方案对比

| 特性 | BigRec_FELLRec | RecFormer_FELLRec |
|------|---------------|-------------------|
| 基础模型 | LLaMA | LongFormer |
| 推荐方式 | 生成式（文本输出物品名） | 传统协同过滤（embedding相似度） |
| 数据格式 | JSON指令格式 | 用户-物品序列 |

### 7. 训练效率优化

1. **8-bit量化**: `load_in_8bit=True`
2. **混合精度训练**: `fp16=True`
3. **梯度累积**: `gradient_accumulation_steps=16`
4. **分布式训练**: 使用HuggingFace Accelerate
