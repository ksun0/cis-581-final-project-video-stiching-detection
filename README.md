# Video Stitching and Detection
Repository for our CIS581 Final Project with Professor Jianbo Shi, Fall 2020.

Srisa Changolkar, Celestina Saven, Vinay Senthil, Kevin Sun (Team 02)

### Description of Code and Layout
The root directory has the `requirements.txt` for required libraries and packages to install. `video-stitching-detection.py` contains code to stitch videos into a singular image, as well perform object detection on it. The accompanying notebook `video-sitching.ipynb` contains intermediate results, explanations, and visualizations for the video stitching segment. The `object-detection.ipynb` contains the process for object detection.

The `data/` folder contains all the input videos, as well as intermediate results and final output images. For example, stitching `data/desk-left-right.mp4` puts intermediate results in `data/desk-left-right/`, and outputs the stiched result in `data/desk-left-right.png`. Then, object detection, will output the stitched result with object labels and rectangles in `data/desk-left-right-detected.png`.

### Installation and Setting Up
1. `pip install -r requirements.txt`
2. `python video-stitching-detection.py`
- this python file will run four stitches from the 4 commands at the bottom, e.g. `video_stitch_optical_flow('data/desk-left-right.mp4', 'data/desk-left-right', 0, 15, 390, 'data/desk-left-right.png')`
- the first argument is the path to the video file to stitch together
- the second arguemnt is the path to the folder to store/generate the static frames to stitch together
- the third argument is the rotation (e.g. if drone is flying left to right, rotation is 0, but if the drone is flying bottom to up, rotation is 270)
- the fourth argument is the number of frames so skip for each static frame to stitch
- the fifth argument is the last frame to process
- the sixth argument is the output path of the final stitched image.
- Finally, object detection code will run on `data/desk-left-right.png`. since we trained our model on desk setups, outputting the detected results in `data/left-right-detected.png`.

---

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* CIS581 Instructors, TA's, and Piazza
