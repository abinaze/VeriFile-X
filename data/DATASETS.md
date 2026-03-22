# VeriFile-X Dataset Registry

## Real Photo Sources
| Name | Images | Size | URL | License |
|------|--------|------|-----|---------|
| RAISE-1k | 1,000 | 3GB | http://loki.disi.unitn.it/RAISE/ | Free research |
| RAISE-8k | 8,156 | 85GB | http://loki.disi.unitn.it/RAISE/ | Free research |
| COCO 2017 val | 5,000 | 1GB | http://images.cocodataset.org/zips/val2017.zip | CC BY 4.0 |
| FFHQ | 70,000 | 90GB | https://github.com/NVlabs/ffhq-dataset | CC BY-NC 2.0 |
| Unsplash Lite | 25,000 | 3GB | https://github.com/unsplash/datasets | Unsplash license |
| Open Images v7 | 9M | streaming | https://storage.googleapis.com/openimages/web/index.html | CC BY 4.0 |
| DIV2K | 1,000 | 7GB | https://data.vision.ee.ethz.ch/cvl/DIV2K/ | Free research |
| Flickr30K | 31,783 | 10GB | https://shannon.cs.illinois.edu/DenotationGraph/ | Free research |

## AI-Generated Sources
| Name | Images | Size | URL | Generator | License |
|------|--------|------|-----|-----------|---------|
| CIFAKE | 120,000 | 1.2GB | https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images | SD 1.4 | CC BY 4.0 |
| GenImage | 1.3M | 200GB | https://github.com/GenImage-Dataset/GenImage | 8 generators | CC BY-NC |
| DiffusionDB | 14M | subset | https://huggingface.co/datasets/poloclub/diffusiondb | SD 1.4/2.0 | CC BY 4.0 |
| JourneyDB | 4M | subset | https://journeydb.github.io | Midjourney v5 | CC BY-NC |
| TPDNE archive | 70,000 | 8GB | https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces | StyleGAN2 | Free |
| CNNDetection | 362,000 | 45GB | https://github.com/peterwang512/CNNDetection | 11 GANs | Free research |
| ArtiFact | 964,000 | 120GB | https://github.com/awsaf49/artifact | 26 generators | CC BY-NC |
| AIGI NeurIPS2023 | 300,000 | 40GB | https://github.com/Ekko-zn/AIGC-Forensics | 14 generators | Free research |
| FaceForensics++ | frames | 16GB | https://github.com/ondyari/FaceForensics | DeepFake | Free research |
| Fake-vs-Real Faces | 140,000 | 2.4GB | https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces | StyleGAN/ProGAN | CC0 |

## Target Split
- Train: 80,000 real + 80,000 AI = 160,000
- Val:   10,000 real + 10,000 AI = 20,000
- Test:  10,000 real + 10,000 AI = 20,000
- Total: 200,000 balanced images

## Quality Rules for Real Photos
- Must have EXIF with camera make/model OR verified raw source (RAISE)
- No editing software in EXIF
- Minimum 256x256 pixels
- JPEG quality >= 50

## Quality Rules for AI Images
- Must have verified generator label
- No camera EXIF
- Minimum 256x256 pixels
- Known source dataset (not random scrape)
