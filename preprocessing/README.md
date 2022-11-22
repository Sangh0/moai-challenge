### preprocessing

- convert dcm to png format [code](https://github.com/yujinkim1/moai-challenge/tree/develop/preprocessing/dcm2png_converter.py)

```
$ python3 dcm2png_converter.py --path ./dataset_dir/
```

- visualize images and labels [code](https://github.com/yujinkim1/moai-challenge/tree/develop/preprocessing/visualize.py)
- should be run on jupyter notebook as following:

```python
from visualize import load_dataset, visualize

Config = {
    'path': './dataset'
    'count': 'How many number of images will you see?'
}

images, labels = load_dataset(Config['path'])
visualize(images, labels, Config['count'])
```