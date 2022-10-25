import torch
assert torch.__version__.startswith("1.8") 

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

import matplotlib.pyplot as plt
import os, cv2, random, json
import numpy as np
class initializeWeights:

    def __init__(self, train_name, workers, batch_size, base_lr, gamma, steps, max_iter, class_num):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"))
        self.cfg.DATASETS.TRAIN = (train_name,) 
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = workers # Çalışan Sayısı
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml") # Ağırlıkları Çeker ve Yapılandırma Dosyasına Ekler
        self.cfg.SOLVER.IMS_PER_BATCH = batch_size  # Batch Size
        self.cfg.SOLVER.BASE_LR = base_lr # Learning Rate (Öğrenme Oranı)
        self.cfg.SOLVER.GAMMA = gamma # Learning Rate Azaltma Çarpımı
        self.cfg.SOLVER.STEPS = steps # Learning Rate Azaltma Adım Sayısı
        self.cfg.SOLVER.MAX_ITER = max_iter # İterasyon Sayısı
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_num # Sınıf Sayısı

    def train_predicts(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True) # Model Sonucu İçin Klasör Oluşturur
        self.trainer = DefaultTrainer(self.cfg) # Modeli Train Moduna Geçirir Yapılandırma Dosyası ile Birlikte
        self.trainer.resume_or_load(resume=False) # Model Eğitimine 0'dan Başlamak İçin False Yapıyoruz
        self.trainer.train()
        evaluator = COCOEvaluator('my_test', self.cfg, False, output_dir="./output/")
        self.cfg.MODEL.WEIGHTS= os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        val_loader = build_detection_test_loader(self.cfg, 'my_test')
        inference_on_dataset(self.trainer.model, val_loader, evaluator)


        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7# Test için Eşik Değerimiz
        self.cfg.DATASETS.TEST = ("my_test", ) # Tets Verilerimiz Yapılandırma Dosyasına Kaydeder
        self.predictor = DefaultPredictor(self.cfg) 

    def shows(self):
        for d in random.sample(self.test_dataset_dicts, 4):    
            img = cv2.imread(d["file_name"])
            outputs = self.predictor(img)
            v = Visualizer(img[:, :, ::-1],
                        metadata=self.microcontroller_metadata, 
                        scale=0.8, 
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.figure(figsize = (20, 10))
            plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
            plt.show()
    

        
def get_data_dicts(directory):

    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        for idx, v in enumerate(img_anns.values()):
          record = {}
          
          filename = os.path.join(directory, img_anns["imagePath"])
          
          record["file_name"] = filename
          record["image_id"] = idx # evaluator parameter
          record["height"] = 1080 # if you have different image sizes you should change them
          record["width"] = 1920

          annos = img_anns["shapes"]
          objs = []
          for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']] 
            poly = [(x, y) for x, y in zip(px, py)] 
            poly = [p for x in poly for p in x]
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0, # for 1 class
                "segmentation": [poly],
                "iscrowd": 0
            }
            objs.append(obj)
          record["annotations"] = objs
          dataset_dicts.append(record)
    return dataset_dicts


classes = ['building']
data_path = r'C:\Users\nisan\Desktop\Nisa_Neretva\Segmentation\Build_Segmentation\build-segmentation-with-detectron2\data_building'
wei_dataPath = r'C:\Users\nisan\Desktop\Nisa_Neretva\Segmentation\Build_Segmentation\build-segmentation-with-detectron2\output\model_final.pth'

for d in ["train", "test"]:
    DatasetCatalog.register(
        "my_" + d, 
        lambda d=d: get_data_dicts(data_path+d)
    )
    MetadataCatalog.get("my_" + d).set(thing_classes=classes)

microcontroller_metadata = MetadataCatalog.get("my_train")
weights = initializeWeights(wei_dataPath, data_path, "my_train", "my_test")
weights.train_predicts()
test_metadata = MetadataCatalog.get("my_test")
test_dataset_dicts = get_data_dicts(data_path+'test', classes)
weights.shows(data_path)
