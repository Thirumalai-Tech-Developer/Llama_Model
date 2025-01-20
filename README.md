
# 🦙 LLama Model 🦙

Welcome to the **LLama Model** repository! 🎉 This is a PyTorch implementation of a transformer-based autoregressive text generation model. It’s lightweight, efficient, and packed with cool features like **rotary positional embeddings** and **grouped-query attention**. Whether you're here to generate text, experiment with transformer architectures, or just explore, you're in the right place! 🚀

---

## 🌟 Features

- **Rotary Positional Embeddings**: 🌀 Enhances the model's ability to understand sequence positions.
- **Grouped-Query Attention**: 🤝 Reduces memory usage by sharing key-value heads across multiple queries.
- **Efficient Autoregressive Decoding**: ⚡ Supports caching for faster text generation.
- **Customizable Configuration**: 🛠️ Adjust model dimensions, layers, and more to fit your needs.

---

## 🏗️ Architecture

The **LLama** model is a **decoder-only transformer** with the following components:

1. **Input Embedding**: 🌐 Maps token IDs to dense vectors.
2. **Decoder Layers**:
   - **Multi-Head Self-Attention**: 🧠 Uses rotary embeddings and grouped-query attention.
   - **Feed-Forward Network**: 🚀 A two-layer network with **SiLU** and **GELU** activations.
   - **Residual Connections**: 🔗 Combines outputs with skip connections and **RMSNorm**.
3. **Projection Layer**: 🎯 Maps the decoder output to the vocabulary size for token prediction.

---

## 🛠️ Installation

To get started, make sure you have **PyTorch** installed. If not, install it using:

```bash
pip install torch
```

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/your-username/llama.git
cd llama
```

---

## 🚀 Usage

### 1. Import the Model

```python
from llama import LLama, ModelArgs

# Initialize model configuration
config = ModelArgs()
model = LLama(config)
```

### 2. Generate Text

```python
# Example input token IDs
input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

# Generate text
generated_ids = model.generate(
    input_ids,
    start_pos=0,
    max_len=20,
    temperature=0.7,
    top_k=50,
    eos_token_id=2
)

print(generated_ids)
```

### 3. Forward Pass

```python
# Forward pass through the model
output = model(input_ids, start_pos=0)
```

---

## ⚙️ Configuration

The `ModelArgs` class allows you to customize the model's hyperparameters. Here are the default values:

```python
class ModelArgs:
    bos_token_id = 1  # Beginning-of-sequence token ID
    eos_token_id = 2  # End-of-sequence token ID
    d_model: int = 48  # Embedding dimension
    hidden_dim: int = 1024  # Hidden dimension for feed-forward network
    seq_len: int = 8192 * 2  # Maximum sequence length
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 72  # Number of decoder layers
    n_kv_heads: int = 3  # Number of key-value heads
    d_ff: int = 48  # Intermediate dimension in feed-forward network
    max_batch_len: int = 8  # Maximum batch size
    vocab_size: int = 2000  # Vocabulary size
    dropout: float = 0.1  # Dropout rate
```

---

## 🧪 Examples

### Example 1: Text Generation

```python
config = ModelArgs()
model = LLama(config)

input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
generated_ids = model.generate(input_ids, start_pos=0, max_len=20, temperature=0.7, top_k=50, eos_token_id=2)
print(generated_ids)
```

### Example 2: Custom Configuration

```python
config = ModelArgs(
    d_model=64,
    n_heads=8,
    n_layers=48,
    vocab_size=5000
)
model = LLama(config)
```

---

## 📜 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute it as you see fit. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by the **LLaMA** architecture from **Meta AI**. 🦙
- Uses **Rotary Positional Embeddings** for improved sequence modeling. 🌀

---

## 💬 Get in Touch

Have questions or suggestions? Feel free to open an issue or reach out to me! Let’s build something awesome together! 🚀

