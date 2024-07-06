<h2 align="center">FreeCompose: <br> Generating Diverse Storytelling Images with Minimal Human Effort</h2>
  <p align="center">
    <a href="https://github.com/Aziily"><strong>Zhekai Chen*</strong></a>
    ·
    <a href="https://github.com/encounter1997"><strong>Wen Wang*</strong></a>
    ·
    <a href="https://github.com/LeoYuan0111"><strong>Zhen Yang</strong></a>
    ·
    <a href="https://github.com/LeoYuan0111"><strong>Zeqing Yuan</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=FaOqRpcAAAAJ"><strong>Hao Chen</strong></a>
    ·
    <a href="https://cshen.github.io/"><strong>Chunhua Shen</strong></a>
    <br>
    Zhejiang University
    <br>
    </br>
        <!-- <a href="https://arxiv.org/abs/2311.11243">
        <img src='https://img.shields.io/badge/arxiv-AutoStory-blue' alt='Paper PDF'></a>
        <a href="https://aim-uofa.github.io/AutoStory/">
        <img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a> -->
  </p>
</p>

<p align="center"><b>Code will be released soon!</b></p>

<div style="padding-left:50px; padding-right:50px;">

<center>
    <image src="./assets/teaser.png" style="width: 600px">
</center>

We offer a novel approach to image composition, which integrates multiple input images into a single, coherent image. Rather than concentrating on specific use cases such as appearance editing (image harmonization) or semantic editing (semantic image composition), we showcase the potential of utilizing the powerful generative prior inherent in large-scale pre-trained diffusion models to accomplish generic image composition applicable to both scenarios.

We observe that the pre-trained diffusion models automatically identify simple copy-paste boundary areas as low-density regions during denoising. Building on this insight, we propose to optimize the composed image towards high-density regions guided by the diffusion prior. In addition, we introduce a novel mask-guided loss to further enable flexible semantic image composition.

Extensive experiments validate the superiority of our approach in achieving generic zero-shot image composition. Additionally, our approach shows promising potential in various tasks, such as object removal and multi-concept customization.

## Method

<center>
    <image src="./assets/method_overview.png" style="width: 800px">
</center>

FreeCompose overview. Our FreeCompose pipeline consists of three phases: object removal, image harmonization, and semantic image composition. In each phase, the pipeline takes an input image and two text prompts to calculate the loss. In the object removal phase, an additional mask is required to select K, V values. In the semantic image composition phase, text prompts can be replaced by other formats, and an additional K, V replacement is implemented for identity consistency.

## Demos

### Object Removal

<center>
    <image src="./assets/removal_res.png" style="width: 800px">
</center>

### Imgae Harmonization

<center>
    <image src="./assets/harmonization_res.png" style="width: 800px">
</center>

### Semantic Image Composition

<center>
    <image src="./assets/composition_res.png" style="width: 800px">
</center>

### SDXL Plug-and-Play

<center>
    <image src="./assets/sdxl.png" style="width: 800px">
</center>

## Citing

If you find our work useful, please consider citing:

<!-- ```BibTeX
@article{AutoStory,
  title={AutoStory: Generating Diverse Storytelling Images with Minimal Human Effort},
  author={Wang, Wen and Zhao, Canyu and Chen, Hao and Chen, Zhekai and Zheng, Kecheng and Shen, Chunhua},
  publisher={arXiv preprint arXiv:2311.11243},
  year={2023},
}
``` -->

</div>
