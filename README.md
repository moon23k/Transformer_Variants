## Transformer Ablation
<br>
Transformer introduced a new approach to sequence processing through the Attention Mechanism, revolutionizing the traditional sequential data processing methods. Along with its success, many research studies based on Transformer has conducted. However, most of these studies focused on utilizing Transformer as it is and exploring additional advancements, resulting in a relatively limited number of studies comparing the performance of natural language processing based on the structural changes of the Transformer model itself. 

To mend this situation, this repo focuses on structure of the Transformer and implements three Transformer models: **Standard Transformer**, **Recurrent Transformer**, and **Evolved Transformer**. The performance evaluation of each model is conducted in three natural language generation tasks: **Neural Machine Translation**, **Dialogue Generation**, and **Text Summarization**. 

<br><br>


## Model Architectures
<br><table>
  <tr align='center'>
    <td><b>Standard Transformer</b></td>
    <td><b>Recurrent Transformer</b></td>
    <td><b>Evolved Transformer</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/moon23k/Transformer_Archs/assets/71929682/5d118ff7-7d8d-4093-ba73-e131e703f467" height=90%; /></td>
    <td><img src="https://github.com/moon23k/Transformer_Archs/assets/71929682/a3a4de91-ecef-4841-a005-c1810ec850ef" /></td>
    <td><img src="https://github.com/moon23k/Transformer_Archs/assets/71929682/1ce4619d-fa7c-44d2-8498-ec568c27994e" /></td>
  </tr>  
</table>
<br><br>


## Experimental Setups
<br>

| &emsp; **Data Setup**                      | &emsp; **Model Setup**                | &emsp; **Training Setup** |
| :---                                       | :---                                  | :---                      |
| **`Machine Translation:`** &hairsp; `WMT14 En-De` | **`Embedding Dimension:`** `256` | **`Epochs:`** `10` |
| **`Dialogue Generation:`** &hairsp; `Daily Dialogue` | **`Hidden Dimension:`** `256`  | **`Batch Size:`** `32` |
| **`Text Summarization:`** &hairsp; `Daily Mail` | **`PFF Dimension:`** `512` | **`Learning Rate:`** `5e-4` |
| **`Train Data Volumn:`** &hairsp; `30,000` | **`N Heads:`** `512` | **`iters_to_accumulate:`** `4`           |
| **`Valid Data Volumn:`** &hairsp; `1,000` | **`N Layers:`** `6` | **`Gradient Clip Max Norm:`** `1` |
| **`Vocab Size:`** `30,000`        | **`N Cells:`** `3`             | **`Apply AMP:`** `True`       |

<br>To shorten the training speed, three techiques are used. <br> 
* **Pre Tokenization** <br>
* **Accumulative Loss Update**, as shown in the table above, accumulative frequency has set 4. <br>
* **Application of AMP**, which enables to convert float32 type vector into float16 type vector.

<br><br>

## Result
<p align='center'>
  <img src="https://github.com/moon23k/Transformer_Archs/assets/71929682/b0d2e7e5-422c-401e-b924-850b7b0b5811" width='80%' height='80%' />
</p>
<br><br>


## How to Use

```
git clone https://github.com/moon23k/Transformer_Arhcs.git
```

```
python3 setup.py
```

```
python3 run.py -mode ['train', 'test', 'inference'] -model ['standard', 'recurrent', 'evolved']
```
<br><br>

## Reference
* [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762) <br>
* [**Universal Transformers**](https://arxiv.org/abs/1807.03819) <br>
* [**The Evolved Transformer**](https://arxiv.org/abs/1901.11117) <br>
