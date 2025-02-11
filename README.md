# README: GANs & Diffusion Models

## StyleGAN2 & StyleGAN3

Emphasizing the **W latent space**, StyleGAN2 & 3 convert an image into a **512-dimensional latent vector**. Image inversion allows the transformation of real images into this latent space, enabling edits and fine-tuning. The models showcase high-quality image synthesis, but also exhibit challenges in **maintaining fine details and global coherence**.

### Results & Observations
- StyleGAN2 & 3 can achieve **very high-resolution images**, but they struggle with **global consistency in complex scenes**.
- **Inversion techniques** attempt to project real images into latent space, but may **lose fine details or distort features**.

| ![Image A](https://github.com/robilovric/DeepFakeGen/blob/main/stylegan3/results/seed4605.png) | ![Image B](https://github.com/robilovric/DeepFakeGen/blob/main/stylegan3/results/seed6399.png) |
|---|---|
| ![Image C](https://github.com/robilovric/DeepFakeGen/blob/main/stylegan3/results/seed6402.png) | ![Image D](https://github.com/robilovric/DeepFakeGen/blob/main/stylegan3/results/seed6401.png) |

### Descriptions
- **Image A**: Demonstrates the abstract backgrounds usually associated with a GAN-generated image and other deviations.
- **Image B**: Exhibits issues that earrings often present for GANs. GANs sometimes have problems with symmetry, particularly earrings.
- **Image C**: Exhibits issues that earrings often present for GANs and contains an abstract background. 
- **Image D**: Contains secondary image and other distorted details.

---

## StyleCLIP: Text-Guided GAN Editing

StyleCLIP enables **text-based modifications** of GAN-generated images. It bridges **GANs and CLIP (Contrastive Language-Image Pretraining)**, leveraging natural language descriptions to steer image synthesis. This effectively introduces **text-to-image capabilities** within the GAN framework.

### Inference & Inversion

| ![Image good inversion](https://github.com/robilovric/DeepFakeGen/blob/main/styleclip/results/dicaprio-real-life-format.jpg) | ![Image bad inversion](https://github.com/robilovric/DeepFakeGen/blob/main/styleclip/results/dicaprio-training-set-format.jpg) |
|---|---|
| **Left: A well-inverted image.** The face closely matches the dataset's expected format, leading to an accurate reconstruction. | **Right: A poorly inverted image.** The original does not align well with the training data, causing distortions. |

### **Why Inversion Struggles in Some Cases**
- **Mismatch with training data:** GANs and diffusion models are trained mostly on **cropped, centered face images**. When given an image with a **different pose, lighting, or visible torso**, the inversion process struggles to map it correctly into latent space.  
- **Information loss:** Since the generator was never trained to reconstruct full-body or off-center images, it **hallucinates missing details**, often leading to **unnatural distortions**.  
- **Latent space constraints:** The model forces all inputs into a specific **512-dimensional latent representation**, which may not fully capture the complexity of real-world images outside its dataset.  



## Stable Diffusion: Robust Text-to-Image Generation

Stable Diffusion takes a **different approach**, using diffusion models for **prompt-driven image synthesis**. Unlike GANs, it iteratively refines **random noise into a coherent image**. It excels at **text-guided image generation**, producing diverse and high-quality outputs.

## GANs vs. Diffusion Models

| Feature | GANs | Diffusion Models |
| --- | --- | --- |
| **Training Stability** | Requires careful tuning | More stable |
| **Image Quality** | High-resolution, but artifacts possible | High-quality and consistent |
| **Diversity** | Limited due to mode collapse | High diversity |
| **Editability** | Strong with latent vector manipulation | Less flexible but powerful |
| **Text-Guided Generation** | StyleCLIP enables text control | Natively supports text prompts |

## The Future: Hybrid GAN-Diffusion Approaches

Combining **GANs and diffusion models** could yield the best of both worlds. A hybrid approach could leverage **GANs' speed and efficiency** with **diffusion models' flexibility and quality**, leading to breakthroughs in **realistic and controllable image synthesis**.

## Citations

(Add references here)
