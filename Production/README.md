# CarPoseML

CarPoseML is a machine learning architecture for estimating the position and orientation (pose) of multiple cars in 
static images.

The output will be a .jpg image with scaled and rotated 3D bounding boxes around all cars in the input .jpg image.

## Installation

Create a python virtual environment (venv) in the same directory as the CarPoseML files.
Activate the venv
Install dependendencies using the `requirements.txt` file

## Usage
From inside the .venv directory, run:

```bash
python local_lambda.py -i <input.jpg> -o <output.jpg>
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
