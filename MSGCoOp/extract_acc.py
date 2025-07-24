import os
import re
from glob import glob
from collections import defaultdict

def extract_accuracy(log_path):
    """从日志文件中提取accuracy"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        match = re.search(r'\* accuracy: (\d+\.\d+)%', content)
        if match:
            return float(match.group(1))
    except:
        pass
    return None

def collect_model_results(root_dir, target_model):
    """收集指定模型在所有数据集上的结果，z按seed分组"""
    results = {
        'base': defaultdict(list),  # 使用列表存储多个seed的结果
        'new': defaultdict(list),
        'datasets': set()
    }
    # 查找所有base训练的log文件
    base_logs = glob(os.path.join(root_dir, '**/train_base/**/log.txt'), recursive=True)
    for log_path in base_logs:
        parts = log_path.split(os.sep)
        dataset = parts[-6]
        model = parts[-4]
        
        if model != target_model:
            continue
            
        accuracy = extract_accuracy(log_path)
        if accuracy is not None:
            results['base'][dataset].append(accuracy)
            results['datasets'].add(dataset)
    
    # 查找所有new测试的log文件
    new_logs = glob(os.path.join(root_dir, '**/test_new/**/log.txt'), recursive=True)
    for log_path in new_logs:
        parts = log_path.split(os.sep)
        dataset = parts[-6]
        model = parts[-4]
        
        if model != target_model:
            continue
            
        accuracy = extract_accuracy(log_path)
        if accuracy is not None:
            results['new'][dataset].append(accuracy)
            results['datasets'].add(dataset)
    
    return results

def calculate_harmonic_mean(base, new):
    """计算调和平均数"""
    if base == 0 or new == 0:
        return 0
    return 2 * base * new / (base + new)

def calculate_average(values):
    """计算平均值"""
    if not values:
        return None
    return sum(values) / len(values)

def print_model_results(results, model_name):
    """打印指定模型在所有数据集上的结果（平均所有seed）"""
    datasets = sorted(results['datasets'])
    
    # 准备数据用于计算总体平均值
    base_sum = 0
    new_sum = 0
    valid_datasets = 0
    
    print(f"\nResults for model: {model_name}")
    print(f"{'Dataset':<15} {'Base':<10} {'New':<10} {'H':<10} {'Seeds':<10}")
    print("-" * 60)
    
    for dataset in datasets:
        base_accs = results['base'].get(dataset, [])
        new_accs = results['new'].get(dataset, [0.0, 0.0, 0.0])
        
        if base_accs and new_accs:
            avg_base = calculate_average(base_accs)
            avg_new = calculate_average(new_accs)
            h = calculate_harmonic_mean(avg_base, avg_new)
            
            # 获取seed数量（取base和new中较小的seed数）
            num_seeds = min(len(base_accs), len(new_accs))
            
            print(f"{dataset:<15} {avg_base:.2f}{'':<6} {avg_new:.2f}{'':<6} {h:.2f}{'':<6} {num_seeds}")
            
            base_sum += avg_base
            new_sum += avg_new
            valid_datasets += 1
    
    # 计算并打印总体平均值
    if valid_datasets > 0:
        avg_base = base_sum / valid_datasets
        avg_new = new_sum / valid_datasets
        avg_h = calculate_harmonic_mean(avg_base, avg_new)
        print("-" * 60)
        print(f"{'Average':<15} {avg_base:.2f}{'':<6} {avg_new:.2f}{'':<6} {avg_h:.2f}")
    else:
        print("No complete dataset results found for this model.")

def main():
    root_dir = 'output_xda'  # 修改为你的output目录路径
    target_model = 'MSGCoOp'  # 指定要分析的模型
    
    results = collect_model_results(root_dir, target_model)
    print_model_results(results, target_model)

if __name__ == '__main__':
    main()

