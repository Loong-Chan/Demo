# Demo
 A code template for implementing small-scale semi-supervised node classification with GNNs.



## Usage

### 1. For replicating the results

```
cd impl/YourModelName
python Main.py --dataset cora --device cuda:0
```

### 2. For fine-tuning the hyper-parameters

``` 
cd impl/YourModelName
bash tunr.sh --dataset cora --device cuda:0
```





