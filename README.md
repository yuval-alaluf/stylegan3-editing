# Third Time's the Charm? Image and Video Editing with StyleGAN3

> Yuval Alaluf*, Or Patashnik*, Zongze Wu, Asif Zamir, Eli Shechtman, Dani Lischinski, Daniel Cohen-Or  
> *Denotes equal contribution  
> 
> StyleGAN is arguably one of the most intriguing and well-studied generative models, demonstrating impressive performance in image generation, inversion, and manipulation. In this work, we explore the recent StyleGAN3 architecture, compare it to its predecessor, and investigate its unique advantages, as well as drawbacks. In particular, we demonstrate that while StyleGAN3 can be trained on unaligned data, one can still use aligned data for training, without hindering the ability to generate unaligned imagery. Next, our analysis of the disentanglement of the different latent spaces of StyleGAN3 indicates that the commonly used W/W+ spaces are more entangled than their StyleGAN2 counterparts, underscoring the benefits of using the StyleSpace for fine-grained editing. Considering image inversion, we observe that existing encoder-based techniques struggle when trained on unaligned data. We therefore propose an encoding scheme trained solely on aligned data, yet can still invert unaligned images. Finally, we introduce a novel video inversion and editing workflow that leverages the capabilities of a fine-tuned StyleGAN3 generator to reduce texture sticking and expand the field of view of the edited video.

<a href=""><img src="https://img.shields.io/badge/arXiv-b31b1b.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>  

<p align="center">
<img src="docs/teaser.jpg" width="800px"/>  
<br>
Using the recent StyleGAN3 generator, we edit unaligned input images across various domains using off-the-shelf editing techniques. Using a trained StyleGAN3 encoder, these techniques can likewise be used to edit real images and videos.
</p>

# Description   
Official implementation of our StyleGAN3 paper "Third Time's the Charm?" where we analyze the recent StyleGAN3 generator and explore its advantages over previous style-based generators. We evaluate StyleGAN3's latent spaces, explore their editability, and introduce an encoding scheme for inverting and editing real images and videos. 

# Code
We're working hard to make this code as accessible and easy to use as possible. Code, editing directions, and pre-trained encoders will be made available shortly. 