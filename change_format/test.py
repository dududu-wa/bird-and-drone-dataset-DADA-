"""
验证YOLO格式数据集标注的可视化工具
随机选择图片并在窗口中显示带边界框的图像
同时支持可视化yolotococo.py生成的COCO格式数据集
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

def visualize_yolo_annotations(data_dir, num_samples=3):
    """
    随机选择指定数量的图片，绘制YOLO格式标注的边界框，并在窗口中显示
    
    参数:
        data_dir: YOLO格式数据集目录
        num_samples: 要处理的图片数量，默认为3
    """
    # 获取所有图像文件
    image_files = [f for f in os.listdir(data_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 没有足够的图片时的处理
    if not image_files:
        print(f"在 {data_dir} 中未找到图像文件")
        return
    
    # 随机选择指定数量的图片
    if len(image_files) < num_samples:
        print(f"警告: 目录中只有 {len(image_files)} 张图片，少于请求的 {num_samples} 张")
        selected_files = image_files
    else:
        selected_files = random.sample(image_files, num_samples)
    
    print(f"随机选择的图片: {selected_files}")
    
    # 尝试读取类别文件
    classes = []
    classes_path = os.path.join(data_dir, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            classes = f.read().strip().split("\n")
        print(f"读取到 {len(classes)} 个类别: {classes}")
        print("注意：YOLO格式中类别ID从0开始，COCO格式中类别ID从1开始")
    
    # 设置随机颜色
    colors = np.random.uniform(0, 255, size=(len(classes) if classes else 100, 3))
    
    # 创建一个大图，用于显示所有选择的图片
    plt.figure(figsize=(15, 5 * num_samples))
    
    # 处理选中的图片
    for i, filename in enumerate(selected_files):
        # 图像文件路径
        image_path = os.path.join(data_dir, filename)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像文件: {image_path}")
            continue
        
        # 图像高度和宽度
        height, width, _ = image.shape
        
        # 对应的标注文件路径
        base_name = os.path.splitext(filename)[0]
        annotation_path = os.path.join(data_dir, f"{base_name}.txt")
        
        # 如果标注文件存在，绘制边界框
        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # 确保有足够的数据
                    class_id = int(parts[0])
                    # 为显示COCO ID，在可视化中标出对应的COCO类别ID
                    coco_class_id = class_id + 1
                    
                    # YOLO格式: 类别ID, 中心点X, 中心点Y, 宽度, 高度 (归一化坐标)
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_width = float(parts[3]) * width
                    box_height = float(parts[4]) * height
                    
                    # 计算左上角坐标
                    x1 = int(x_center - box_width / 2)
                    y1 = int(y_center - box_height / 2)
                    # 计算右下角坐标
                    x2 = int(x_center + box_width / 2)
                    y2 = int(y_center + box_height / 2)
                    
                    # 确保坐标在图像范围内
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    
                    # 选择颜色
                    color = colors[class_id % len(colors)]
                    color = (int(color[0]), int(color[1]), int(color[2]))
                    
                    # 绘制边界框
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # 如果有类别列表，添加类别名称标签
                    if classes and class_id < len(classes):
                        # 显示COCO格式的类别ID和名称
                        label = f"{classes[class_id]}(ID:{coco_class_id})"
                        # 添加类别标签
                        cv2.putText(image, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 在Matplotlib中显示图像
        plt.subplot(num_samples, 1, i+1)
        plt.title(f"图像 {i+1}: {filename}")
        # OpenCV的BGR转为RGB用于matplotlib显示
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()  # 显示图像，不保存
    
    # 同时在OpenCV窗口中显示每张图片
    for i, filename in enumerate(selected_files):
        image_path = os.path.join(data_dir, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            continue
            
        # 对应的标注文件路径
        base_name = os.path.splitext(filename)[0]
        annotation_path = os.path.join(data_dir, f"{base_name}.txt")
        
        # 图像高度和宽度
        height, width, _ = image.shape
        
        # 如果标注文件存在，绘制边界框
        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    coco_class_id = class_id + 1
                    
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_width = float(parts[3]) * width
                    box_height = float(parts[4]) * height
                    
                    x1 = int(x_center - box_width / 2)
                    y1 = int(y_center - box_height / 2)
                    x2 = int(x_center + box_width / 2)
                    y2 = int(y_center + box_height / 2)
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    color = colors[class_id % len(colors)]
                    color = (int(color[0]), int(color[1]), int(color[2]))
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    if classes and class_id < len(classes):
                        label = f"{classes[class_id]}(ID:{coco_class_id})"
                        cv2.putText(image, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 在OpenCV窗口中显示
        window_name = f"图像 {i+1}: {filename}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)  # 调整窗口大小
        cv2.imshow(window_name, image)
    
    print("按任意键关闭OpenCV窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"完成了 {len(selected_files)} 张图片的可视化处理。")

def visualize_coco_dataset(coco_dir, num_samples=3, dataset_type="train"):
    """
    可视化COCO格式数据集中的图像和标注框
    
    参数:
        coco_dir: COCO数据集的根目录，包含annotations和图像目录
        num_samples: 要可视化的图像数量
        dataset_type: 数据集类型，可以是"train"、"val"或"test"
    """
    print(f"正在从 {coco_dir} 读取COCO格式数据集...")
    
    # 构建路径
    annotations_path = os.path.join(coco_dir, "annotations", f"instances_{dataset_type}.json")
    images_dir = os.path.join(coco_dir, dataset_type)
    
    # 检查路径是否有效
    if not os.path.exists(annotations_path):
        print(f"未找到标注文件: {annotations_path}")
        return
    
    if not os.path.exists(images_dir):
        print(f"未找到图像目录: {images_dir}")
        return
    
    # 读取COCO格式的标注文件
    try:
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"读取标注文件失败: {e}")
        return
    
    # 检查数据结构
    if not all(key in coco_data for key in ["images", "annotations", "categories"]):
        print("标注文件格式不正确，缺少必要的字段")
        return
    
    print(f"找到 {len(coco_data['images'])} 张图像, {len(coco_data['annotations'])} 个标注, {len(coco_data['categories'])} 个类别")
    
    # 创建图像ID到文件名的映射
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # 创建图像ID到标注的映射
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(ann)
    
    # 创建类别ID到类别名称的映射
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 随机选择图像
    if len(coco_data['images']) < num_samples:
        print(f"警告: 数据集中只有 {len(coco_data['images'])} 张图片，少于请求的 {num_samples} 张")
        selected_images = coco_data['images']
    else:
        selected_images = random.sample(coco_data['images'], num_samples)
    
    # 设置随机颜色
    colors = np.random.uniform(0, 255, size=(len(category_id_to_name) + 1, 3))
    
    # 创建matplotlib图形
    plt.figure(figsize=(15, 5 * len(selected_images)))
    
    for i, img_info in enumerate(selected_images):
        image_id = img_info['id']
        filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # 构建图像路径
        image_path = os.path.join(images_dir, filename)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        
        # 检查图像尺寸是否与标注匹配
        if image.shape[1] != img_width or image.shape[0] != img_height:
            print(f"警告: 图像 {filename} 的尺寸与标注不匹配")
        
        # 获取该图像的所有标注
        annotations = image_id_to_annotations.get(image_id, [])
        
        # 在图像上绘制边界框
        for ann in annotations:
            category_id = ann['category_id']
            bbox = ann['bbox']  # [x, y, width, height]
            
            # 获取类别名称
            category_name = category_id_to_name.get(category_id, f"未知类别 {category_id}")
            
            # 计算边界框坐标
            x, y, w, h = [int(v) for v in bbox]
            
            # 选择颜色
            color = colors[category_id % len(colors)]
            color = (int(color[0]), int(color[1]), int(color[2]))
            
            # 绘制边界框
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # 添加类别标签
            label = f"{category_name} (ID:{category_id})"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 添加到matplotlib图形
        plt.subplot(len(selected_images), 1, i + 1)
        plt.title(f"图像 {i+1}: {filename} (ID: {image_id})")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # 同时在OpenCV窗口中显示
        window_name = f"COCO图像 {i+1}: {filename}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        cv2.imshow(window_name, image)
    
    plt.tight_layout()
    plt.show()
    
    print("按任意键关闭OpenCV窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"完成了 {len(selected_images)} 张COCO图像的可视化处理")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='数据集可视化工具')
    parser.add_argument('--mode', type=str, choices=['yolo', 'coco'], default='yolo',
                       help='可视化模式: yolo或coco (默认: yolo)')
    parser.add_argument('--dir', type=str, default="C:/Users/TR/Desktop/bird",
                       help='数据集目录')
    parser.add_argument('--samples', type=int, default=3,
                       help='要可视化的样本数量 (默认: 3)')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test'], default='train',
                       help='COCO模式下要可视化的数据集类型 (默认: train)')
    
    args = parser.parse_args()
    
    if args.mode == 'yolo':
        print(f"正在可视化YOLO格式数据集: {args.dir}")
        visualize_yolo_annotations(args.dir, args.samples)
    else:
        print(f"正在可视化COCO格式数据集: {args.dir}")
        visualize_coco_dataset(args.dir, args.samples, args.dataset)
    
    print("可视化完成。")
