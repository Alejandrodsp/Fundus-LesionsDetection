# Improved Fundus Lesions Detection
<img src="https://user-images.githubusercontent.com/54383618/211225454-be653f9c-60aa-499e-abc4-c5e19e026a8a.gif" width="400" height="400"/>
A new method that uses a YOLOR-CSP architecture combined with the SAHI framework to improve detection of fundus lesions.



### 🔧 Installation
```
git clone https://github.com/Alejandrodsp/Fundus-Lesions-Detection
cd Fundus-Lesions-Detection
pip install -r requirements.txt
```
- [Download the proposed work with SGD here](https://github.com/Alejandrodsp/Fundus-Lesions-Detection/releases/download/proposed-work-SGD/best.pt)
- [Download the proposed work with Adam here](https://github.com/Alejandrodsp/Fundus-Lesions-Detection/releases/download/proposed-work-Adam/best.pt)
- [Download pre-training weights here](https://github.com/Alejandrodsp/Fundus-Lesions-Detection/releases/download/pre-training/yolor_csp.pt)
### ⚙️ Pre-processing steps
1. **Cropping**
```
cd pre-processing/cropping
python main.py
cd ..
cd ..
```
2. **Get .json with bounding box annotations**
```
cd utils/globox/globox
python convertYoloToCoco.py
cd ..
cd ..
cd ..
```
3. **SAHI**
```
cd pre-processing/sahi
sahi coco slice ../../datasets/DDR-CROPPING/train/images ../../datasets/DDR-CROPPING/train/_annotations.coco.json --output_dir ../../datasets/DDR-CROPPING-SAHI/train
sahi coco slice ../../datasets/DDR-CROPPING/valid/images ../../datasets/DDR-CROPPING/valid/_annotations.coco.json --output_dir ../../datasets/DDR-CROPPING-SAHI/valid
sahi coco slice ../../datasets/DDR-CROPPING/test/images ../../datasets/DDR-CROPPING/test/_annotations.coco.json --output_dir ../../datasets/DDR-CROPPING-SAHI/test
cd ..
cd ..
```
4. **Generate annotations in .txt**

- Access https://www.makesense.ai/.
- Click on button "Get Started".
- Select all images from **DDR-CROPPING-SAHI/test/_annotations.coco_images_512_02**.
- Click on **Object Detection** button and next in **Start Project**.
- Click on the **Actions** button and then click on **Import annotations** and then click on **Single file in COCO JSON format** and select the **_annotations.coco_512_02.json** file from the DDR-CROPPING-SAHI/test folder, and then click on **Import** button.
- Click on the **Actions** button and then click on **Export annotations** and then click on **A .zip package containing files in YOLO format**.
- Create a folder named **labels** inside DDR-CROPPING-SAHI/test and extract the .zip generated there.
- Rename the folder **_annotations.coco_images_512_02** to **images** and the file **_annotations.coco_512_02.json** to **_annotations.coco.json**.
- Repeat the same steps for the **train** and **valid** sets.

### :hourglass_flowing_sand: Training
```
cd model/yolor
python train.py --batch-size 16 --img 640 640 --data data.yaml --cfg cfg/yolor_csp.cfg --weights yolor_csp.pt --device 0 --hyp hyp.scratch.640.yaml --epochs 8000 --noautoanchor
```
### :heavy_check_mark: Inference
```
cd model/yolor
python detect.py --source ../../datasets/DDR/test/images --cfg cfg/yolor_csp.cfg --weights best.pt --conf 0.25 --img-size 640 --device 0
```

### :chart_with_upwards_trend: Testing
```
cd model/yolor
python test.py --data data/data.yaml --iou 0.45 --weights best.pt --batch 16 --img 640 --cfg cfg/yolor_csp.cfg --verbose
```

### ✒️ Authors
* **Alejandro Pereira** - Computer Science, Federal University of Pelotas - UFPel, Pelotas, Brazil.
* **Carlos Santos** - Federal Institute of Education, Science and Technology Farroupilha - IFFar, Alegrete, Brazil.
* **Marilton Aguiar** - Postgraduate Program in Computing, Federal University of Pelotas - UFPel, Pelotas, Brazil.
* **Daniel Welfer** - Departament of Applied Computing, Federal University of Santa Maria - UFSM, Santa Maria, Brazil.
* **Marcelo Dias** - Computer Science, Federal University of Pelotas - UFPel, Pelotas, Brazil.
* **Marcelo Ribeiro** - Computer Science, Federal University of Pelotas - UFPel, Pelotas, Brazil.

### :punch: Acknowledgements
* https://github.com/WongKinYiu/yolor
* https://github.com/obss/sahi
* https://github.com/laclouis5/globox
* https://github.com/SkalskiP/make-sense
* Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES).
