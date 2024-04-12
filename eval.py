from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances

import cv2
import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
import torch

from ubteacher import add_ubteacher_config
from ubteacher.engine.source_fft_np_trainer import UBTeacherTrainer, BaselineTrainer
from ubteacher.modeling.meta_arch.cp_rcnn import Ori_TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.meta_arch.two_head_rcnn_refine import Two_head_TwoStagePseudoLabGeneralizedRCNN_REFINE
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin_69
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.roi_heads.roi_heads_two_head_refine_relation import StandardROIHeadsPseudoLab_object_relation
from ubteacher.engine.source_fft_np_trainer_two_head_version2_object_relation import Two_head_fft_UBTeacherTrainer_V2_object_relation

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes


def single_image_dataset():
    image_path = "/app/datasets/bdd100k/val/b1cd1e94-26dd524f.jpg"
    # image_path = "/app/datasets/bdd100k/val/b2d502aa-ef17ffbd.jpg"
    # image_path = "/app/datasets/bdd100k/val/b2e2f4ed-6ba045d0.jpg"
    return [{"file_name": image_path, "image_id": 0, "height": 720, "width": 1280}]


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def predict_image(args):
    cfg = setup(args)

    cfg.defrost()  # "разморозка" конфига
    cfg.DATASETS.TRAIN = ("single_image",)
    cfg.DATASETS.TEST = ()

    cfg.freeze()  # "заморозка" конфига обратно


    image_path = "/app/datasets/bdd100k/val/b1cd1e94-26dd524f.jpg"
    # image_path = "/app/datasets/bdd100k/val/b2d502aa-ef17ffbd.jpg"
    # image_path = "/app/datasets/bdd100k/val/b2e2f4ed-6ba045d0.jpg"
    output_path = '/app/datasets/result_image1.jpg'
    
    Trainer = Two_head_fft_UBTeacherTrainer_V2_object_relation
    model = Trainer.build_model(cfg)
    model_teacher = Trainer.build_model(cfg)
    ensem_ts_model = EnsembleTSModel(model_teacher, model)

    DetectionCheckpointer(
        ensem_ts_model, save_dir=cfg.OUTPUT_DIR
    ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    ensem_ts_model.eval()

    # Применяем модель к изображению
    image = cv2.imread(image_path)
    print(f"Размер изображения: {image.shape}")
    image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    batched_inputs = [{"image": image_tensor}]

    with torch.no_grad():
        outputs = model(batched_inputs)
        print(f"outputs: {outputs}")
        
    # Получаем первый список из outputs, который содержит предсказания
    first_output_list = outputs[0]

    # Теперь, предполагая, что каждый элемент в этом списке - это словарь, содержащий 'instances',
    # обращаемся к первому элементу (словарю) в списке
    first_output_dict = first_output_list[0]

    # Получаем объект 'instances' из словаря
    instances = first_output_dict["instances"]

    print(f"instances: {instances}")
    pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
    print(f"Предсказанные рамки (до фильтрации): {pred_boxes}")
    pred_classes = instances.pred_classes.cpu().numpy()
    pred_scores = instances.scores.cpu().numpy()

    # Фильтрация по порогу
    threshold = 0.8
    selected_indices = pred_scores > threshold
    pred_boxes = pred_boxes[selected_indices]
    pred_classes = pred_classes[selected_indices]
    pred_scores = pred_scores[selected_indices]

    filtered_instances = Instances(image.shape[:2])
    filtered_instances.pred_boxes = Boxes(pred_boxes)
    filtered_instances.pred_classes = pred_classes
    filtered_instances.scores = pred_scores

    # Визуализируем результаты
    visualizer = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    vis_output = visualizer.draw_instance_predictions(predictions=filtered_instances)
    result_image = vis_output.get_image()[:, :, ::-1]

    # Сохраняем изображение
    cv2.imwrite(output_path, result_image)

    return outputs


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    DatasetCatalog.register("single_image", single_image_dataset)
    MetadataCatalog.get("single_image").set(thing_classes=[])  # Поскольку у нас нет разметки, список классов пуст

    launch(
        predict_image,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )