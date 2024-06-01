# SplitAnotationData
 アノテーションをした画像の分割

```
python SplitAnotationData.py [dir_path]

```

|name|description|default|
|:-:|:-:|:-:|
|dir_path|Directory containing path to split||
|--save_path|Directory to save the results|results/{nowtime}|
|--trainrate/-train|train rate|0.7|
|--valrate/-val|val rate|0.1|
|--testrate/-test|test rate|0.2|
|--ext|extension|jpg|
