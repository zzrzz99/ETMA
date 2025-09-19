#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click run all face restoration methods

Automatically execute the following steps:
1. Image preprocessing (face detection and cropping)
2. Parallel processing of four restoration methods
3. Result post-processing and merging
"""

import argparse
import os
import sys
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from shared_utils.utils import setup_logging, create_output_dir
from shared_utils.preprocess import preprocess_images


def run_defakehop(input_dir: str, output_dir: str, **kwargs) -> dict:
    """运行DefakeHop恢复"""
    try:
        from defakehop.restore import DefakeHopRestorer
        
        restorer = DefakeHopRestorer(device=kwargs.get('device', 'auto'))
        results = restorer.restore_batch(input_dir, output_dir)
        
        logging.info(f"DefakeHop完成: {results}")
        return {"method": "DefakeHop", "results": results, "status": "success"}
        
    except Exception as e:
        logging.error(f"DefakeHop运行失败: {e}")
        return {"method": "DefakeHop", "error": str(e), "status": "failed"}


def run_neural_texture(input_dir: str, output_dir: str, **kwargs) -> dict:
    """运行Neural Texture Reversion恢复"""
    try:
        from neural_texture_reversion.restore import NeuralTextureRestorer
        
        restorer = NeuralTextureRestorer(device=kwargs.get('device', 'auto'))
        results = restorer.restore_batch(input_dir, output_dir)
        
        logging.info(f"Neural Texture Reversion完成: {results}")
        return {"method": "Neural Texture Reversion", "results": results, "status": "success"}
        
    except Exception as e:
        logging.error(f"Neural Texture Reversion运行失败: {e}")
        return {"method": "Neural Texture Reversion", "error": str(e), "status": "failed"}


def run_face2face(input_dir: str, output_dir: str, **kwargs) -> dict:
    """运行Face2Face恢复"""
    try:
        from reverting_face2face.restore import Face2FaceRestorer
        
        restorer = Face2FaceRestorer(device=kwargs.get('device', 'auto'))
        results = restorer.restore_batch(input_dir, output_dir)
        
        logging.info(f"Face2Face恢复完成: {results}")
        return {"method": "Face2Face", "results": results, "status": "success"}
        
    except Exception as e:
        logging.error(f"Face2Face恢复运行失败: {e}")
        return {"method": "Face2Face", "error": str(e), "status": "failed"}


def run_faceswap(input_dir: str, output_dir: str, **kwargs) -> dict:
    """运行FaceSwap恢复"""
    try:
        from df_restore_gan.restore import FaceSwapRestorer
        
        restorer = FaceSwapRestorer(device=kwargs.get('device', 'auto'))
        results = restorer.restore_batch(input_dir, output_dir)
        
        logging.info(f"FaceSwap恢复完成: {results}")
        return {"method": "FaceSwap", "results": results, "status": "success"}
        
    except Exception as e:
        logging.error(f"FaceSwap恢复运行失败: {e}")
        return {"method": "FaceSwap", "error": str(e), "status": "failed"}


def run_all_methods(input_dir: str, output_dir: str, **kwargs) -> Dict[str, dict]:
    """
    运行所有恢复方法
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        **kwargs: 其他参数
        
    Returns:
        Dict[str, dict]: 各方法的运行结果
    """
    # 创建输出目录
    create_output_dir(output_dir)
    
    # 定义所有恢复方法
    methods = [
        ("defakehop", run_defakehop),
        ("neural_texture", run_neural_texture),
        ("face2face", run_face2face),
        ("faceswap", run_faceswap)
    ]
    
    results = {}
    
    # 串行运行（避免GPU内存冲突）
    for method_name, method_func in methods:
        method_output_dir = os.path.join(output_dir, method_name)
        logging.info(f"开始运行 {method_name}...")
        
        start_time = time.time()
        result = method_func(input_dir, method_output_dir, **kwargs)
        end_time = time.time()
        
        result["execution_time"] = end_time - start_time
        results[method_name] = result
        
        logging.info(f"{method_name} 运行完成，耗时: {end_time - start_time:.2f}秒")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="一键运行所有人脸恢复方法")
    parser.add_argument("--input_dir", required=True, help="输入图像目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--preprocess", action="store_true", help="是否进行预处理")
    parser.add_argument("--face_size", type=int, default=512, help="人脸尺寸")
    parser.add_argument("--margin", type=float, default=0.2, help="边界扩展比例")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"], help="计算设备")
    parser.add_argument("--log_level", default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return 1
    
    start_time = time.time()
    
    # 预处理阶段
    if args.preprocess:
        logger.info("开始图像预处理...")
        preprocess_dir = os.path.join(args.output_dir, "preprocessed")
        
        preprocess_results = preprocess_images(
            args.input_dir,
            preprocess_dir,
            args.face_size,
            args.margin
        )
        
        if preprocess_results["success"] == 0:
            logger.error("预处理失败，没有成功处理的图像")
            return 1
        
        logger.info(f"预处理完成: {preprocess_results}")
        input_dir = preprocess_dir
    else:
        input_dir = args.input_dir
    
    # 运行所有恢复方法
    logger.info("开始运行所有恢复方法...")
    results = run_all_methods(input_dir, args.output_dir, device=args.device)
    
    # 输出结果统计
    total_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info("所有恢复方法运行完成！")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info("=" * 50)
    
    success_count = 0
    failed_count = 0
    
    for method_name, result in results.items():
        if result["status"] == "success":
            success_count += 1
            logger.info(f"✅ {method_name}: 成功")
        else:
            failed_count += 1
            logger.error(f"❌ {method_name}: 失败 - {result.get('error', '未知错误')}")
    
    logger.info(f"成功: {success_count}/{len(results)} 个方法")
    logger.info(f"失败: {failed_count}/{len(results)} 个方法")
    
    # 生成结果报告
    report_path = os.path.join(args.output_dir, "run_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("人脸恢复运行报告\n")
        f.write("=" * 30 + "\n")
        f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total_time:.2f}秒\n")
        f.write(f"输入目录: {args.input_dir}\n")
        f.write(f"输出目录: {args.output_dir}\n\n")
        
        for method_name, result in results.items():
            f.write(f"{method_name}:\n")
            if result["status"] == "success":
                f.write(f"  状态: 成功\n")
                f.write(f"  执行时间: {result['execution_time']:.2f}秒\n")
                f.write(f"  处理结果: {result['results']}\n")
            else:
                f.write(f"  状态: 失败\n")
                f.write(f"  错误: {result.get('error', '未知错误')}\n")
            f.write("\n")
    
    logger.info(f"详细报告已保存到: {report_path}")
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
