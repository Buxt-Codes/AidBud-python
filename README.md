# AidBud

This is an offline triaging application that can be used anywhere, anytime. The current android implementation is in the works and is not currently running. This python library serves as a proof-of-concept as well as a starting ground for the main application.

## Features

- Conversations
- Patient Cards
- Toggleable Triaging
- Contextual Information

## How to Use

1. Install the library and its dependencies. 
`pip install "git+https://github.com/Buxt-Codes/AidBud-python.git"`

2. Login into HuggingFace and configure pytorch.
```
from huggingface_hub import login
import os
import torch
import torch._dynamo

os.environ["DISABLE_TORCH_COMPILE"] = "1"
torch.compile = lambda x, *args, **kwargs: x  
torch._dynamo.config.suppress_errors = True
login(token="") # INSERT HUGGING FACE TOKEN HERE
``` 

3. Import AidBud and initialise the model.
```
from aidbud install AidBud
aidbud = AidBud()
aidbud.initialise()
```

4. Start new conversation and query away!
```
aidbud.new_conversation()
aidbud.query("I have a friend who just got burnt on his hand.")
```