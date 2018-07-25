# DeepStudy
Deep learning paper implementation study at the <a href="https://datalab.snu.ac.kr/">DataMining Lab</a> in the <a href="http://www.snu.ac.kr">Seoul National University</a>.

# GoogLeNet
<details><summary>Click to show results</summary>
<p>
## Hyper Parameters
* Loss = CrossEntropyLoss
* Adam Optimizer = learning rate : 1e-3, weight_decay : 5e-4

## Experiments
* Dataset = CIFAR10
### Without BathNormalization
* Epoch 10 => Accuracy 48.89% took 1054 secs(about half hour)
* Epoch 100 => Accuracy 75.62% took 9649 secs(about 2.5 hour)
* Epoch 300 => Accuracy 79.65% took 28728 secs(about 8 hour)
* ~ Epoch 900 => Accuracy 78~81%

### With BathNormalization on every Conv Layers, Learning rate : 1e-3
* Epoch 10 => Accuracy 61.45% took 1149 secs(about half hour)

### With BathNormalization on every Conv Layers, SGD optimizer Learning rate : 1e-2
* Epoch 10 => Accuracy 72.3% took 1159 secs(about half hour)
* Epoch 20 => Accuracy 81.18% took 2213 secs
* Epoch 139 => Accuracy 89.2% took took 14763 secs

### With BathNormalization on every Conv Layers, SGD optimizer Learning rate : 1e-1
* Epoch 10 => Accuracy 56.2% took 1155 secs(about half hour)

### With BathNormalization on every Conv Layers, BatchNorm after Inception, SGD optimizer Learning rate : 1e-2
* Epoch 10 => Accuracy 71.99% took 1159 secs(about half hour)

### With BathNormalization on every Conv Layers, BatchNorm after Inception, SGD optimizer Learning rate : 1e-1
* Epoch 10 => Accuracy 60.69% took 1116 secs(about half hour)
</p>
</details>
