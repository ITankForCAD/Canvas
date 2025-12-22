# Context Compression using 2D Grids
This repository contains a project that compresses context using 2D grids. The main components are:

inference.py: This script demonstrates how to use the trained model for inference. It takes a source text and generates a visual representation of the compressed context using a canvas.

utils.py: Contains utility functions for training the model, computing embeddings, and encoding the context into grids/canvases.


### Run the inference script:


```python
python inference.py
```
This will load the pre-trained model, generate embeddings for the source text, and display the visual representation of the compressed context. You can input any text within the ```source``` variable. Additionally, you can set DEBUG to false if you wish to obtain the true grids rather than the human-optimized grids.

Saving Grids
To save the generated grids for later use, you can modify the inference.py script to include:


```python
torch.save(grids, 'path/to/saved/grids.pt')
```
Then, to load the saved grids:


```python
grids = torch.load('path/to/saved/grids.pt')
```
