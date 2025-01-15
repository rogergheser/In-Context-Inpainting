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
This program will simply generate a set of images for each of the classes in `inputs/` directory. The first image will be used as a background and the rest will be used to guide the foreground inpainting.

1. **Download the Pretrained Model:**
   - Download the pretrained model from [this link](https://pan.baidu.com/s/1HPbRRE5ZtPRpOSocm9qOmA?pwd=BA1c).

2. **Prepare the dataset:**
   Ensure that your ICM-57 is ready.

3. **Run the Evaluation:**
   Use the following command to run the evaluation script. Replace the placeholders with the actual paths if they differ.

   ```bash
   python eval.py --checkpoint PATH_TO_MODEL --save_path results/ --config config/eval.yaml
   ```

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

### Acknowledgments

We would like to express our gratitude to the developers and contributors of the [DIFT](https://github.com/Tsingularity/dift) and [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt/) projects. Their shared resources and insights have significantly aided the development of our work.

## Statement

<!-- If you are interested in our work, please consider citing the following:
```

``` -->

This project is under the MIT license. For technical questions, please contact <strong><i>He Guo</i></strong> at [hguo01@hust.edu.cn](mailto:hguo01@hust.edu.cn). For commerial use, please contact <strong><i>Hao Lu</i></strong> at [hlu@hust.edu.cn](mailto:hlu@hust.edu.cn)
