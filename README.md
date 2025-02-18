# OVERVIEW: GANs and Diffusion Models in DeepFake Generation

## [StyleGAN2](https://github.com/robilovric/DeepFakeGen/blob/main/stylegan2/StyleGAN2.ipynb) & [StyleGAN3](https://github.com/robilovric/DeepFakeGen/blob/main/stylegan3/StyleGAN3.ipynb)
They can generate images by mapping them to a **512-dimensional latent vector** within the **W latent space**, allowing fine-grained **control over attributes**. Image inversion enables real images to be embedded into this space for editing and manipulation. While StyleGAN2 achieves high-quality synthesis, it often struggles with texture sticking and spatial inconsistencies. StyleGAN3 addresses these issues by improving geometric consistency and smooth transitions, making it more suitable for transformations and animations. Despite advancements, both models face challenges in preserving fine details and handling complex backgrounds.

### Results & Observations
StyleGAN2 and 3 can achieve **very high-resolution images**, but they struggle with **global consistency in complex scenes**.

| ![Image A](https://github.com/robilovric/DeepFakeGen/blob/main/stylegan3/results/seed3397.png) | ![Image B](https://github.com/robilovric/DeepFakeGen/blob/main/stylegan3/results/seed6399.png) |
|---|---|
| ![Image C](https://github.com/robilovric/DeepFakeGen/blob/main/stylegan3/results/seed6402.png) | ![Image D](https://github.com/robilovric/DeepFakeGen/blob/main/stylegan3/results/seed6401.png) |

- **Image A**: Demonstrates the abstract backgrounds usually associated with a GAN-generated image and other deviations.
- **Image B**: Shows distorted output, GANs sometimes have big problems with symmetry.
- **Image C**: Exhibits issues that earrings often present for GANs and contains an abstract background. 
- **Image D**: Contains secondary image and other distorted details.


### [More Results](https://github.com/robilovric/DeepFakeGen/tree/main/stylegan3/results)

---

## [StyleCLIP: Text-Guided GAN Editing](https://github.com/robilovric/DeepFakeGen/blob/main/styleclip/StyleCLIP.ipynb)

StyleCLIP enables **text-based modifications** of GAN-generated images. It bridges **GANs and CLIP (Contrastive Language-Image Pretraining)**, leveraging natural language descriptions to steer image synthesis. This effectively introduces **text-to-image capabilities** within the GAN framework.

### **Inversion techniques** 
Attempt to project real images into latent space, but may **lose fine details or distort features**.

### Inversion Steps

| ![Image good inversion](https://github.com/robilovric/DeepFakeGen/blob/main/styleclip/results/affleck.jpg) |
|---|
| **A well-inverted image.** The face closely matches the dataset's expected format, leading to an accurate reconstruction. |
| ![Image bad inversion](https://github.com/robilovric/DeepFakeGen/blob/main/styleclip/results/kerum2.jpg) |
| **A poorly inverted image.** The original does not align well with the training data, causing distortions. |

### **Why Inversion Struggles in Some Cases**
- **Mismatch with training data:** GANs and diffusion models are trained mostly on **cropped, centered face images**. When given an image with a **different pose, lighting, or visible torso**, the inversion process struggles to map it correctly into latent space.  
- **Information loss:** Since the generator was never trained to reconstruct full-body or off-center images, it **hallucinates missing details**, often leading to **unnatural distortions**.  
- **Latent space constraints:** The model forces all inputs into a specific **512-dimensional latent representation**, which may not fully capture the complexity of real-world images outside its dataset.  

| ![Image good inversion](https://github.com/robilovric/DeepFakeGen/blob/main/styleclip/results/kerum-inverted.png) |
|---|
| This image shows the **original image** after exiting the **alignment process**, the output is then used to finalize **inversion process** and get the inverted image. |
| ![Image good inversion](https://github.com/robilovric/DeepFakeGen/blob/main/styleclip/results/kerum-manipulated.png) |
| This image shows a **well-aligned face** that successfully passed the **inversion process**. The **inverted image** is now **treated as the original** for all further manipulations. |

## [More Results](https://github.com/robilovric/DeepFakeGen/tree/main/styleclip/results)

---

## [CycleGAN](https://github.com/robilovric/DeepFakeGen/blob/main/cyclegan/CycleGAN.ipynb)  

CycleGAN is a **Generative Adversarial Network (GAN)** designed for **image-to-image translation** without requiring **paired training data**. Unlike traditional supervised methods that need corresponding image pairs (e.g., a horse and its exact zebra version), CycleGAN learns to **map one domain to another using only unpaired images**.  

Before CycleGAN, most image-to-image translation models relied on **large datasets with paired examples** (e.g., Pix2Pix). However, in many real-world scenarios, obtaining such paired data is **expensive or impractical**.  

CycleGAN introduces a **cycle consistency loss** to ensure that an image translated from domain **A to B** can be **converted back to A** while preserving its original structure. This **bidirectional mapping** ensures that transformations remain faithful to the source image.  

### **Key Features**  
- **Unpaired Image-to-Image Translation**: No need for perfectly aligned training pairs.  
- **Cycle Consistency**: Ensures the model does not introduce arbitrary distortions.  
- **Preserves Content**: Maintains the structure of input images while modifying style.  
- **Applications**: Style transfer, domain adaptation, artistic rendering, medical imaging, and more.   

### Results & Observations
By eliminating the need for paired data and **ensuring structure-preserving transformations**, CycleGAN has become a powerful tool for **unsupervised image translation**, significantly expanding the possibilities of generative AI.

| ![Image Input](https://github.com/robilovric/DeepFakeGen/blob/main/cyclegan/results/n02381460_140_real.png) |  
|---|  
| **Input Image**: A real image of horses in the wild. |  

| ![Image Output](https://github.com/robilovric/DeepFakeGen/blob/main/cyclegan/results/n02381460_140_fake.png) |  
|---|  
| **Output Image**: The same scene, but the horses have been transformed into zebras. Domain translation from horses to zebras. |  

## [More Results](https://github.com/robilovric/DeepFakeGen/tree/main/cyclegan/results)

---

## [Stable Diffusion: Robust Text-to-Image Generation](https://github.com/robilovric/DeepFakeGen/blob/main/stablediffusion/StableDiffusion.ipynb)

Stable Diffusion takes a **different approach**, using diffusion models for **prompt-driven image synthesis**. Unlike GANs, it iteratively refines **random noise into a coherent image**. 

### Results & Observations
It excels at **text-guided image generation**, producing diverse and high-quality outputs.

| ![Image A](https://github.com/robilovric/DeepFakeGen/blob/main/stablediffusion/results/1926283858.png) | ![Image B](https://github.com/robilovric/DeepFakeGen/blob/main/stablediffusion/results/3825872414.png) |
|---|---|
| ![Image C](https://github.com/robilovric/DeepFakeGen/blob/main/stablediffusion/results/1795594041.png) | ![Image D](https://github.com/robilovric/DeepFakeGen/blob/main/stablediffusion/results/3901221944.png) |

Diffusion models excel at generating **highly detailed and realistic images**, often surpassing GANs in fine texture reproduction and global coherence. Their **iterative denoising process** allows for intricate details, smooth shading, and natural variations, making them particularly effective for **photo-realistic portraits and complex scenes**. However, while diffusion shines in **capturing texture and lighting**, it may struggle with **spatial consistency**—sometimes introducing distortions in object structure or logical coherence. Unlike GANs, which can suffer from **mode collapse**, diffusion models provide greater diversity in outputs, making them powerful tools for **high-quality image synthesis and controlled transformations**.

### [More results](https://github.com/robilovric/DeepFakeGen/tree/main/stablediffusion/results/)

---

## GANs vs. Diffusion Models

| **Feature**           | **CycleGAN**                                                                                               | **StyleGAN2 & 3**                                                                                               | **Stable Diffusion**                                                                                               |
|-----------------------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **Objective**         | Unpaired image-to-image translation                                                                        | High-quality image synthesis from latent space                                                                | Text-to-image generation                                                                                           |
| **Training Data**     | Unpaired images                                                                                           | Large datasets of real images                                                                                | Text-image pairs                                                                                                   |
| **Latent Space**      | No explicit latent space; operates directly on images                                                     | Operates in W and W+ spaces                                                                                   | Utilizes a diffusion process in latent space                                                                       |
| **Image Fidelity**    | Effective for style and texture transformations; may struggle with fine details and structural consistency | Produces sharp, coherent images with fine details; may face challenges with diversity                         | Generates highly detailed images; quality depends on prompt specificity                                            |
| **Control**           | Direct domain mapping                                                                                     | Latent vector manipulation allows fine-grained control over image attributes                                  | Control through textual prompts and conditioning                                                                   |
| **Generalization**    | Limited to transformations between trained domains                                                        | Excels in generating variations within the training data distribution                                         | Generalizes well across various styles and concepts                                                                |
| **Training Stability**| Generally stable due to cycle-consistency loss                                                            | Requires careful tuning to prevent issues like mode collapse                                                  | More stable training dynamics                                                                                      |
| **Inference Speed**   | Fast due to direct image transformation                                                                   | Fast image generation from latent vectors                                                                     | Slower due to iterative denoising process                                                                          |
| **Diversity**         | May produce limited diversity if not properly regularized                                                 | Can suffer from mode collapse, leading to less diverse outputs                                                | Capable of producing a wide range of diverse images                                                                |
| **Editability**       | Limited to learned transformations between domains                                                        | High editability through latent space manipulations                                                           | Editing involves modifying textual prompts or conditioning inputs                                                  |
| **Text-Guided Generation** | Not natively supported; requires additional models for text guidance                                 | Not natively supported; extensions like StyleCLIP enable text-based control                                   | Natively supports text-to-image generation through prompt inputs                                                   |

<!-- | Feature | GANs | Diffusion Models |
| --- | --- | --- |
| **Training Stability** | Requires careful tuning | More stable |
| **Image Quality** | High-resolution, but artifacts possible | High-quality and consistent |
| **Diversity** | Limited due to mode collapse | High diversity |
| **Editability** | Strong with latent vector manipulation | Less flexible but powerful |
| **Text-Guided Generation** | StyleCLIP enables text control | Natively supports text prompts | -->

# Final Thoughts on GANs and Diffusion Models

## Overview
Generative Adversarial Networks (GANs) and Diffusion Models have made significant strides in the field of image synthesis. While GANs offer speed and efficiency, diffusion models excel in generating highly detailed and controllable outputs. Through our evaluations, both approaches have demonstrated strengths and weaknesses, raising an important question: **Should future research focus on leveraging high-quality datasets and specialized training, or should we push toward novel, hybrid methods that redefine image generation altogether?**

## Key Findings
### **GANs: Strengths & Limitations**
✅ **Speed & Efficiency** – GANs generate images rapidly after training.
✅ **High-Resolution Outputs** – Well-trained GANs produce photorealistic results.
❌ **Mode Collapse** – GANs sometimes fail to capture the full diversity of the dataset.
❌ **Training Instability** – Requires careful hyperparameter tuning and loss balancing.

### **Diffusion Models: Strengths & Limitations**
✅ **Superior Image Quality** – Step-by-step refinement leads to high realism.
✅ **Better Diversity & Control** – Latent space conditioning enables fine control over outputs.
❌ **Slow Inference** – Multiple denoising steps make generation slower than GANs.
❌ **Computationally Expensive** – Requires significant resources for training and inference.

## Future Directions
### **Path 1: Specialized Training with High-Quality Datasets**
- Fine-tune models on domain-specific datasets (e.g., medical imaging, artistic style transfer).
- Improve dataset curation and annotation for richer feature learning.
- Optimize model architectures to reduce artifacts and bias in generated content.

### **Path 2: Knowledge Breakthrough – Bridging GANs & Diffusion**
- Develop hybrid models that combine GANs' speed with diffusion's controllability.
- Explore novel architectures beyond current paradigms (e.g., diffusion-powered GANs).
- Introduce **self-supervised learning** techniques to enhance generalization across tasks.

## Conclusion
Both GANs and diffusion models have revolutionized image generation, but the next frontier lies in either **perfecting specialized models** through better datasets and training strategies or **breaking new ground** by merging their capabilities into a unified, next-gen generative model. Whether the future leans toward targeted applications or a fundamental shift in generative AI, one thing is clear: the evolution of image synthesis is far from over.

---
**Next Steps:** Conduct further experiments on hybrid architectures, dataset augmentation, and model efficiency improvements.


## Citations
@inproceedings{Karras2021,
  author = {Tero Karras and Miika Aittala and Samuli Laine and Erik H\"ark\"onen and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  title = {Alias-Free Generative Adversarial Networks},
  booktitle = {Proc. NeurIPS},
  year = {2021}
}

@InProceedings{Patashnik_2021_ICCV,
    author    = {Patashnik, Or and Wu, Zongze and Shechtman, Eli and Cohen-Or, Daniel and Lischinski, Dani},
    title     = {StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2085-2094}
}

@InProceedings{Rombach_2022_CVPR,
    author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
    title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {10684-10695}
}

