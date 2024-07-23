# Image Colorizer

This project is an image colorization application that uses deep learning to convert black and white images into color images. The application is built with Streamlit and employs a pre-trained Convolutional Neural Network (CNN) to achieve realistic colorization results.

## About the Project

This project leverages the technique presented in Zhang et al.'s 2016 ECCV paper, "[Colorful Image Colorization](https://richzhang.github.io/colorization/)." Developed at the University of California, Berkeley by Richard Zhang, Phillip Isola, and Alexei A. Efros, this approach uses Convolutional Neural Networks to infer the colors of a grayscale image.

Previous methods of colorizing black and white images relied heavily on manual annotations, often resulting in desaturated and less believable colorizations. Zhang et al.'s method, however, treats the problem as a classification task, embracing the uncertainty of colorization and enhancing the diversity of colors in the final result.

The neural network was trained on the ImageNet dataset, converting images from the RGB color space to the Lab color space. The L channel (lightness) is used as input, while the a and b channels (color information) are predicted by the network.

## Requirements

To run this project, you will need the following libraries and tools:

- Python 3.7+
- numpy
- opencv-python
- streamlit
- pillow

You can install these dependencies using the following command:
```bash
pip install numpy opencv-python streamlit pillow
```

Additionally, you will need the pre-trained model files:
- `colorization_release_v2.caffemodel`
- `colorization_deploy_v2.prototxt`
- `pts_in_hull.npy`

Place these files in a folder named `models` in the project directory. [Download](https://drive.google.com/drive/folders/1yBmHV7E51CQN5lTp07_pHGzVfC46Rihg?usp=sharing) the files.

## Project Breakdown

### colorize_image(img)
This function performs the colorization of a grayscale image. It follows these steps:
1. Convert the input image to grayscale and then back to RGB.
2. Load the pre-trained neural network and cluster centers.
3. Prepare the image by scaling and converting it to the Lab color space.
4. Perform colorization by predicting the ab channels using the neural network.
5. Combine the L channel with the predicted ab channels and convert the result back to RGB.

### Streamlit UI
The Streamlit UI provides a user-friendly interface for the image colorization application. Key components include:
- A file uploader to allow users to upload black and white images.
- Display of the original and colorized images side by side.
- Custom CSS styling to enhance the visual appearance of the application.

## How to Run the Project

1. Clone the repository.
2. Place the pre-trained model files in the `models` folder.
3. Install the required dependencies.
4. Run the Streamlit application:
```bash
streamlit run app.py
```
5. Open your web browser and navigate to `http://localhost:8501` to use the application.

## Reference

The technique used in this project is based on the following paper:
- Zhang, R., Isola, P., & Efros, A. A. (2016). Colorful Image Colorization. In ECCV.

This project uses a CNN to predict the color channels of a grayscale image. The network was trained on the ImageNet dataset, converting images from the RGB to the Lab color space. The L channel is used as input, and the network predicts the a and b channels. The final colorized image is obtained by combining the L channel with the predicted a and b channels and converting back to the RGB color space.

## Acknowledgements

- Richard Zhang, Phillip Isola, and Alexei A. Efros for their groundbreaking work on image colorization.
- The developers and contributors of the libraries and tools used in this project.
