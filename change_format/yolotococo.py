"""
YOLO 格式的数据集转化为 COCO 格式的数据集
--root_dir 输入根路径
--save_path 保存文件的名字(没有random_split时使用)
--random_split 有则会随机划分数据集，然后再分别保存为3个文件。
--split_by_file 按照 ./train.txt ./val.txt ./test.txt 来对数据集进行划分。
"""

import os
import json
import cv2
from tqdm import tqdm
import random
import shutil


def rename_files_sequentially(yolo_dir):
    """
    按顺序重命名所有图像文件和对应的标注文件
    格式为: 00001.jpg, 00001.txt, 00002.jpg, 00002.txt, ...
    """
    print("开始按顺序重命名文件...")
    # 获取所有图像文件
    image_files = [f for f in os.listdir(yolo_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 创建临时映射字典保存原始文件名到新文件名的映射
    file_mapping = {}
    
    for i, filename in enumerate(image_files, 1):
        # 原始文件路径
        image_path = os.path.join(yolo_dir, filename)
        base_name = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]
        
        # 对应的标注文件
        txt_file = f"{base_name}.txt"
        txt_path = os.path.join(yolo_dir, txt_file)
        
        # 新的文件名
        new_base_name = f"{i:05d}"  # 5位数字，如00001
        new_image_name = f"{new_base_name}{extension}"
        new_txt_name = f"{new_base_name}.txt"
        
        # 保存映射关系，避免直接重命名导致的冲突
        file_mapping[filename] = new_image_name
        if os.path.exists(txt_path):
            file_mapping[txt_file] = new_txt_name
    
    # 创建临时目录用于存放重命名的文件
    temp_dir = os.path.join(yolo_dir, "_temp_rename")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # 复制到临时目录并重命名
    for old_name, new_name in tqdm(file_mapping.items()):
        old_path = os.path.join(yolo_dir, old_name)
        temp_path = os.path.join(temp_dir, new_name)
        
        if os.path.exists(old_path):
            shutil.copy2(old_path, temp_path)
    
    # 确认所有文件都已复制后，将临时文件移回原目录
    for file in os.listdir(temp_dir):
        temp_path = os.path.join(temp_dir, file)
        dest_path = os.path.join(yolo_dir, file)
        shutil.move(temp_path, dest_path)
    
    # 删除临时目录
    os.rmdir(temp_dir)
    
    # 删除原始文件（只删除那些已经重命名的文件）
    for old_name in file_mapping.keys():
        old_path = os.path.join(yolo_dir, old_name)
        if old_name not in file_mapping.values() and os.path.exists(old_path):
            os.remove(old_path)
    
    print(f"文件重命名完成：共处理 {len(image_files)} 个图像文件和对应的标注文件")
    
    # 返回新的文件列表（只包含图像文件）
    return [f for f in file_mapping.values() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]


def yolo_to_coco(yolo_dir, output_dir, split_ratio=(0.7, 0.2, 0.1), rename_files=True):
    # 如果需要，先对文件进行重命名
    if rename_files:
        renamed_image_files = rename_files_sequentially(yolo_dir)
        # 使用重命名后的文件列表
        image_files = renamed_image_files
    else:
        # 获取所有图像文件
        image_files = [filename for filename in os.listdir(yolo_dir) 
                    if filename.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 初始化 COCO 格式的字典
    coco_train = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    coco_val = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    coco_test = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 读取类别名称
    with open(os.path.join(yolo_dir, "classes.txt"), "r") as f:
        classes = f.read().strip().split("\n")

    # 填充类别信息 - 类别ID从1开始
    for i, cls in enumerate(classes):
        category = {
            "id": i + 1,  # 从1开始而不是0
            "name": cls,
            "supercategory": cls  # 使用类别名称作为supercategory
        }
        coco_train["categories"].append(category)
        coco_val["categories"] = coco_train["categories"]
        coco_test["categories"] = coco_train["categories"]

    # 创建标准COCO目录结构
    annotations_dir = os.path.join(output_dir, "annotations")
    train_images_dir = os.path.join(output_dir, "train")
    val_images_dir = os.path.join(output_dir, "val")
    test_images_dir = os.path.join(output_dir, "test")
    
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)
    if not os.path.exists(train_images_dir):
        os.makedirs(train_images_dir)
    if not os.path.exists(val_images_dir):
        os.makedirs(val_images_dir)
    if not os.path.exists(test_images_dir):
        os.makedirs(test_images_dir)
    
    # 更明确的随机分配方式
    random.seed(42)  # 设置随机种子，保证可重复性
    total_files = len(image_files)
    train_size = int(total_files * split_ratio[0])
    val_size = int(total_files * split_ratio[1])
    
    # 随机选择训练集
    train_files = random.sample(image_files, train_size)
    
    # 从剩余文件中随机选择验证集
    remaining_files = list(set(image_files) - set(train_files))
    val_files = random.sample(remaining_files, val_size)
    
    # 剩下的作为测试集
    test_files = list(set(remaining_files) - set(val_files))
    
    print(f"数据集已随机分配: 训练集 {len(train_files)}张, 验证集 {len(val_files)}张, 测试集 {len(test_files)}张")

    def process_files(files, coco_data, images_dir, start_image_id=0, start_annotation_id=0):
        image_id = start_image_id
        annotation_id = start_annotation_id
        for filename in tqdm(files):
            image_path = os.path.join(yolo_dir, filename)
            img = cv2.imread(image_path)
            height, width, _ = img.shape

            # 生成新的文件名并确保以jpg结尾
            new_filename = f"{image_id:012d}.jpg"
            dst_image_path = os.path.join(images_dir, new_filename)
            
            # 如果原图不是jpg格式，需要转换
            if not filename.lower().endswith('.jpg'):
                cv2.imwrite(dst_image_path, img)
            else:
                shutil.copy2(image_path, dst_image_path)

            # 添加图像信息
            coco_data["images"].append({
                "id": image_id,
                "file_name": new_filename,
                "width": width,
                "height": height,
                "license": 1,  # 默认许可证ID
                "flickr_url": "",  # 可选的URL字段
                "coco_url": "",  
                "date_captured": ""  # 捕获日期字段
            })

            # 对应的 YOLO 标注文件路径
            yolo_annotation_path = os.path.join(
                yolo_dir, os.path.splitext(filename)[0] + ".txt")

            if os.path.exists(yolo_annotation_path):
                with open(yolo_annotation_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                        
                    yolo_category_id = int(parts[0])
                    coco_category_id = yolo_category_id + 1  # YOLO的0对应COCO的1
                    
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    # 转换为 COCO 格式的边界框 - 左上角坐标，宽和高
                    x = (x_center - w / 2) * width
                    y = (y_center - h / 2) * height
                    box_width = w * width
                    box_height = h * height
                    
                    # 确保坐标和尺寸不为负值
                    x = max(0, x)
                    y = max(0, y)
                    box_width = max(1, min(width - x, box_width))
                    box_height = max(1, min(height - y, box_height))

                    # 计算面积
                    area = box_width * box_height

                    # 添加标注信息
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": coco_category_id,  # 修正为coco_category_id
                        "bbox": [x, y, box_width, box_height],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": []  # 空分割数据，使用矩形框
                    }
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1

            image_id += 1
        return coco_data, image_id, annotation_id

    print("处理训练集...")
    coco_train, next_image_id, next_annotation_id = process_files(
        train_files, coco_train, train_images_dir)
    print("处理验证集...")
    coco_val, next_image_id, next_annotation_id = process_files(
        val_files, coco_val, val_images_dir, next_image_id, next_annotation_id)
    print("处理测试集...")
    coco_test, _, _ = process_files(
        test_files, coco_test, test_images_dir, next_image_id, next_annotation_id)

    # 保存为 COCO 格式的 JSON 文件
    print("保存标注文件...")
    with open(os.path.join(annotations_dir, "instances_train.json"), "w") as f:
        json.dump(coco_train, f)
    with open(os.path.join(annotations_dir, "instances_val.json"), "w") as f:
        json.dump(coco_val, f)
    with open(os.path.join(annotations_dir, "instances_test.json"), "w") as f:
        json.dump(coco_test, f)
    
    print(f"转换完成！数据集已保存到 {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='将YOLO格式数据集转换为COCO格式')
    parser.add_argument('--yolo_dir', type=str, default="C:/Users/TR/Desktop/drone",
                        help='YOLO数据集目录路径')
    parser.add_argument('--output_dir', type=str, default="C:/Users/TR/Desktop/coco_dataset",
                        help='输出COCO数据集的目录路径')
    parser.add_argument('--split_ratio', type=str, default="0.7,0.2,0.1",
                        help='训练集、验证集、测试集的划分比例，用逗号分隔')
    parser.add_argument('--no_rename', action='store_true',
                        help='不对文件进行重命名处理')
    
    args = parser.parse_args()
    
    # 解析划分比例
    split_ratio = [float(x) for x in args.split_ratio.split(',')]
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    yolo_to_coco(args.yolo_dir, args.output_dir, split_ratio, rename_files=not args.no_rename)