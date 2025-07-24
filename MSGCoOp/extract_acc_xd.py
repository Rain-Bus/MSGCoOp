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

def collect_cross_dataset_results(root_dir, target_model):
    """收集跨数据集测试结果，按DATASET、KG_WEIGHT、LOADEP分组"""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # results[dataset][kg_weight][loadep] = [acc1, acc2, acc3] (for 3 seeds)
    
    # 查找所有测试结果的log文件
    # 路径格式: output_1120_xd/base2new/train_base/{DATASET}/shots_16_{KG_WEIGHT}_ep{LOADEP}/{TRAINER}/{CFG}/seed{SEED}/log.txt
    test_logs = glob(os.path.join(root_dir, '**/train_base/**/log.txt'), recursive=True)
    
    for log_path in test_logs:
        parts = log_path.split(os.sep)
        
        # 解析路径结构
        try:
            # 找到包含shots_和_ep的部分
            shots_part = None
            for part in parts:
                if 'shots_' in part and '_ep' in part:
                    shots_part = part
                    break
            
            if not shots_part:
                continue
                
            # 解析shots_16_{KG_WEIGHT}_ep{LOADEP}
            # 例如: shots_16_0.5_ep50
            pattern = r'shots_\d+_([^_]+)_ep(\d+)'
            match = re.search(pattern, shots_part)
            if not match:
                continue
                
            kg_weight = match.group(1)
            loadep = int(match.group(2))
            
            # 获取数据集名称（shots_part的前一个部分）
            shots_index = parts.index(shots_part)
            dataset = parts[shots_index - 1]
            
            # 获取模型名称
            model = parts[shots_index + 1]
            if model != target_model:
                continue
            
            # 跳过imagenet训练结果，只关注target数据集的测试结果
            if dataset == 'imagenet':
                continue
                
        except (ValueError, IndexError):
            continue
        
        accuracy = extract_accuracy(log_path)
        if accuracy is not None:
            results[dataset][kg_weight][loadep].append(accuracy)
    
    return results

def calculate_average(values):
    """计算平均值"""
    if not values:
        return None
    return sum(values) / len(values)

def print_cross_dataset_results(results, model_name):
    """打印跨数据集测试结果"""
    print(f"\nCross-dataset results for model: {model_name}")
    print("=" * 80)
    
    # 获取所有的kg_weight和loadep值
    all_kg_weights = set()
    all_loadeps = set()
    all_datasets = set(results.keys())
    
    for dataset_results in results.values():
        all_kg_weights.update(dataset_results.keys())
        for kg_results in dataset_results.values():
            all_loadeps.update(kg_results.keys())
    
    all_kg_weights = sorted(all_kg_weights)
    all_loadeps = sorted(all_loadeps)
    all_datasets = sorted(all_datasets)
    
    # 为每个KG_WEIGHT创建一个表格
    for kg_weight in all_kg_weights:
        print(f"\nKG_WEIGHT: {kg_weight}")
        print("-" * 80)
        
        # 表头
        header = f"{'Dataset':<15}"
        for loadep in all_loadeps:
            header += f"{'ep' + str(loadep):<8}"
        header += f"{'Avg':<8}"
        print(header)
        print("-" * 80)
        
        # 用于计算每个epoch的总体平均值
        epoch_totals = defaultdict(list)
        dataset_averages = []
        
        # 为每个数据集打印结果
        for dataset in all_datasets:
            row = f"{dataset:<15}"
            dataset_accs = []
            
            for loadep in all_loadeps:
                accs = results[dataset].get(kg_weight, {}).get(loadep, [])
                if accs:
                    avg_acc = calculate_average(accs)
                    row += f"{avg_acc:.2f}{'':<4}"
                    dataset_accs.append(avg_acc)
                    epoch_totals[loadep].append(avg_acc)
                else:
                    row += f"{'N/A':<8}"
            
            # 计算该数据集在所有epoch上的平均值
            if dataset_accs:
                dataset_avg = calculate_average(dataset_accs)
                row += f"{dataset_avg:.2f}"
                dataset_averages.append(dataset_avg)
            else:
                row += f"{'N/A':<8}"
            
            print(row)
        
        # 打印每个epoch的平均值
        if epoch_totals:
            avg_row = f"{'Avg_all':<15}"
            epoch_avgs = []
            for loadep in all_loadeps:
                if epoch_totals[loadep]:
                    epoch_avg = calculate_average(epoch_totals[loadep])
                    avg_row += f"{epoch_avg:.2f}{'':<4}"
                    epoch_avgs.append(epoch_avg)
                else:
                    avg_row += f"{'N/A':<8}"
            
            # 总体平均值
            if epoch_avgs:
                total_avg = calculate_average(epoch_avgs)
                avg_row += f"{total_avg:.2f}"
            else:
                avg_row += f"{'N/A':<8}"
            
            print("-" * 80)
            print(avg_row)

def print_best_epoch_analysis(results, model_name):
    """分析每个KG_WEIGHT下的最佳epoch"""
    print(f"\n\nBest epoch analysis for model: {model_name}")
    print("=" * 60)
    
    # 获取所有的kg_weight
    all_kg_weights = set()
    for dataset_results in results.values():
        all_kg_weights.update(dataset_results.keys())
    
    all_kg_weights = sorted(all_kg_weights)
    
    for kg_weight in all_kg_weights:
        print(f"\nKG_WEIGHT: {kg_weight}")
        print("-" * 40)
        
        # 计算每个epoch的总体平均准确率
        epoch_averages = defaultdict(list)
        
        for dataset, dataset_results in results.items():
            if kg_weight in dataset_results:
                for loadep, accs in dataset_results[kg_weight].items():
                    if accs:
                        avg_acc = calculate_average(accs)
                        epoch_averages[loadep].append(avg_acc)
        
        # 计算每个epoch的平均值并找到最佳
        epoch_scores = {}
        for loadep, acc_list in epoch_averages.items():
            if acc_list:
                epoch_scores[loadep] = calculate_average(acc_list)
        
        if epoch_scores:
            # 按准确率排序
            sorted_epochs = sorted(epoch_scores.items(), key=lambda x: x[1], reverse=True)
            
            print(f"{'Epoch':<10} {'Avg Accuracy':<15}")
            print("-" * 25)
            for loadep, avg_acc in sorted_epochs[:5]:  # 显示前5个最佳epoch
                print(f"{loadep:<10} {avg_acc:.2f}")

def main():
    root_dir = 'output_xda'  # 修改为你的output目录路径
    target_model = 'MSGCoOp'  # 指定要分析的模型

    results = collect_cross_dataset_results(root_dir, target_model)
    print_cross_dataset_results(results, target_model)
    print_best_epoch_analysis(results, target_model)

if __name__ == '__main__':
    main()

