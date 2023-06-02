# LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning
![Arch_figure](figure/framework.png)
This repository contains PyTorch implementation for our paper：LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning

### Abstract
We introduce a novel OOD detection approach called **Lo**cal regularized **Co**ntext **Op**timization (**LoCoOp**), which performs OOD regularization that utilizes the portions of CLIP local features as OOD features during training. CLIP's local features have a lot of ID-irrelevant nuisances (e.g., backgrounds), and by learning to push them away from the ID class text embeddings, we can remove the nuisances in the ID class text embeddings and enhance the separation between ID and OOD. Experiments on the large-scale ImageNet OOD detection benchmarks demonstrate the superiority of our LoCoOp over zero-shot, fully supervised detection methods and prompt learning methods. Notably, even in one shot setting -- just one label per class, LoCoOp outperforms existing zero-shot and fully supervised detection methods.

## Note
We will publish the code used for training, including each configuration, upon acceptance.  Thank you for your understanding. If you are interested in our work, you can press the "watch" botton to monitor this repository.