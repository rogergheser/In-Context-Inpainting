<h1 align="center">In-Context Inpainting</h1>

DISCLAIMER:
The following is an extension of In-Context Matting [Original repo](https://github.com/tiny-smart/in-context-matting)

<p align="center">
<a href="https://arxiv.org/pdf/2403.15789.pdf"><img  src="demo/src/icon/arXiv-Paper.svg" ></a>
<!-- <a href="https://link.springer.com/article/"><img  src="demo/src/icon/publication-Paper.svg" ></a> -->
<a href="https://opensource.org/licenses/MIT"><img  src="demo/src/icon/license-MIT.svg"></a>

</p>

## Requirements
We follow the environment setup of [In-Context Matting](https://github.com/tiny-smart/in-context-matting) and [Blended Latent Diffusion](https://github.com/omriav/blended-latent-diffusion).

## Usage

To generate images which are the inpainting of different subjects taken in a subset of the ICM-57 dataset, you can simply use`generate.py`

```bash
python generate.py --output_dir results/
```
This program will simply generate a set of images for each of the classes in `inputs/` directory. The first image will be used as a background and the rest will be used to guide the foreground for inpainting.

## Using other objects

If you want to use other objects samples from the `ICM57` dataset, first you need to follow the instruction in [In-Context Matting](https://github.com/tiny-smart/in-context-matting) to download the dataset and the model weights.

1. **Download the Pretrained Model:**
   - Download the pretrained model from [this link](https://pan.baidu.com/s/1HPbRRE5ZtPRpOSocm9qOmA?pwd=BA1c).

2. **Prepare the dataset:**
   Ensure that your ICM-57 is ready following the dataset section.

3. **Run the Evaluation:**
   Use the following command to run the evaluation script. Replace the placeholders with the actual paths if they differ.

   ```bash
   python eval.py --checkpoint PATH_TO_MODEL --save_path results/ --config config/eval.yaml
   ```
Now, you just need to move the results to the `inputs/` directory and run the `generate.py` script.

### Dataset
**ICM-57**
- Download link: [ICM-57 Dataset](https://pan.baidu.com/s/1ZJU_XHEVhIaVzGFPK_XCRg?pwd=BA1c)
- **Installation Guide**:
  1. After downloading, unzip the dataset into the `datasets/` directory of the project.
  2. Ensure the structure of the dataset folder is as follows:
     ```
     datasets/ICM57/
     ├── image
     └── alpha
     ```

## Contributors
- [Amir Gheser](https://github.com/rogergheser)
- [Alessio Pierdominici](https://github.com/EXINEF)
- [Matteo Mascherin](https://github.com/MatteoMaske)

