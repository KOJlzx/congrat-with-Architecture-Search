from transformers import AutoTokenizer, AutoModel
# AutoTokenizer.from_pretrained("distilgpt2")
AutoModel.from_pretrained("distilgpt2", cache_dir="~/.cache/huggingface/hub")